# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging
from multiprocessing.dummy import Array

import dpctl.memory as dpctl_mem
import numpy as np
from numba.core import types

import numba_dpex.utils as utils
from numba_dpex.core.exceptions import (
    SUAIProtocolError,
    UnsupportedAccessQualifierError,
    UnsupportedKernelArgumentError,
)
from numba_dpex.dpctl_iface import USMNdArrayType


class Packer:

    # TODO: Remove after NumPy support is removed
    _access_types = ("read_only", "write_only", "read_write")

    def _check_for_invalid_access_type(self, access_type):
        if access_type not in Packer._access_types:
            raise UnsupportedAccessQualifierError()
        #     msg = (
        #         "[!] %s is not a valid access type. "
        #         "Supported access types are [" % (access_type)
        #     )
        #     for key in self.valid_access_types:
        #         msg += " %s |" % (key)

        #     msg = msg[:-1] + "]"
        #     if access_type is not None:
        #         print(msg)
        #     return True
        # else:
        #     return False

    def _get_info_from_suai(self, obj):
        """
        Extracts the metadata of an arrya-like object that provides a
        __sycl_usm_array_interface__ (SUAI) attribute.

        The ``dpctl.memory.as_usm_memory`` function converts the array-like
        object into a dpctl.memory.USMMemory object. Using the ``as_usm_memory``
        is an implicit way to verify if the array-like object is a legal
        SYCL USM memory back Python object that can be passed to a dpex kernel.

        Args:
            obj: array-like object with a SUAI attribute.

        Returns:
            usm_mem: USM memory object.
            total_size: Total number of items in the array.
            shape: Shape of the array.
            ndim: Total number of dimensions.
            itemsize: Size of each item.
            strides: Stride of the array.
            dtype: Dtype of the array.
        """
        try:
            usm_mem = dpctl_mem.as_usm_memory(obj)
        except Exception:
            logging.exception(
                "array-like object does not implement the SUAI protocol."
            )
            # TODO
            raise SUAIProtocolError()

        shape = obj.__sycl_usm_array_interface__["shape"]
        total_size = np.prod(obj.__sycl_usm_array_interface__["shape"])
        ndim = len(obj.__sycl_usm_array_interface__["shape"])
        itemsize = np.dtype(
            obj.__sycl_usm_array_interface__["typestr"]
        ).itemsize
        dtype = np.dtype(obj.__sycl_usm_array_interface__["typestr"])
        strides = obj.__sycl_usm_array_interface__["strides"]

        if strides is None:
            strides = [1] * ndim
            for i in reversed(range(1, ndim)):
                strides[i - 1] = strides[i] * shape[i]
            strides = tuple(strides)

        return usm_mem, total_size, shape, ndim, itemsize, strides, dtype

    def _unpack_array_helper(self, size, itemsize, buf, shape, strides, ndim):
        """
        Implements the unpacking logic for array arguments.

        TODO: Add more detail

        Args:
            size: Total number of elements in the array.
            itemsize: Size in bytes of each element in the array.
            buf: The pointer to the memory.
            shape: The shape of the array.
            ndim: Number of dimension.

        Returns:
            A list a ctype value for each array attribute argument
        """
        unpacked_array_attrs = []

        # meminfo (FIXME: should be removed and the USMArrayType modified once
        # NumPy support is removed)
        unpacked_array_attrs.append(ctypes.c_size_t(0))
        # meminfo (FIXME: Evaluate if the attribute should be removed and the
        # USMArrayType modified once NumPy support is removed)
        unpacked_array_attrs.append(ctypes.c_size_t(0))
        unpacked_array_attrs.append(ctypes.c_longlong(size))
        unpacked_array_attrs.append(ctypes.c_longlong(itemsize))
        unpacked_array_attrs.append(buf)
        for ax in range(ndim):
            unpacked_array_attrs.append(ctypes.c_longlong(shape[ax]))
        for ax in range(ndim):
            unpacked_array_attrs.append(ctypes.c_longlong(strides[ax]))

        return unpacked_array_attrs

    def _unpack_usm_array(self, val):
        (
            usm_mem,
            total_size,
            shape,
            ndim,
            itemsize,
            strides,
            dtype,
        ) = self._get_info_from_suai(val)

        return self._unpack_array_helper(
            total_size,
            itemsize,
            usm_mem,
            shape,
            strides,
            ndim,
        )

    def _unpack_array(self, val, access_type):
        packed_val = val
        # Check if the NumPy array is backed by USM memory
        usm_mem = utils.has_usm_memory(val)

        # If the NumPy array is not USM backed, then copy to a USM memory
        # object. Add an entry to the repack_map so that on exit from kernel
        # the USM object can be copied back into the NumPy array.
        if usm_mem is None:
            self._check_for_invalid_access_type(access_type)
            usm_mem = utils.as_usm_obj(val, queue=self._queue, copy=False)

            orig_val = val
            packed = False
            if not val.flags.c_contiguous:
                # If the numpy.ndarray is not C-contiguous
                # we pack the strided array into a packed array.
                # This allows us to treat the data from here on as C-contiguous.
                # While packing we treat the data as C-contiguous.
                # We store the reference of both (strided and packed)
                # array and during unpacking we use numpy.copyto() to copy
                # the data back from the packed temporary array to the
                # original strided array.
                packed_val = val.flatten(order="C")
                packed = True

            if access_type == "read_only":
                utils.copy_from_numpy_to_usm_obj(usm_mem, packed_val)
            elif access_type == "read_write":
                utils.copy_from_numpy_to_usm_obj(usm_mem, packed_val)
                # Store to the repack map
                self._repack_map.update(
                    {orig_val: (usm_mem, packed_val, packed)}
                )
            elif access_type == "write_only":
                self._repack_map.update(
                    {orig_val: (usm_mem, packed_val, packed)}
                )

        return self._unpack_array_helper(
            packed_val.size,
            packed_val.dtype.itemsize,
            usm_mem,
            packed_val.shape,
            packed_val.strides,
            packed_val.ndim,
        )

    def _unpack_argument(self, ty, val):
        """
        Unpack a Python object into a ctype value using Numba's
        type-inference machinery.

        Args:
            ty: The data types of the kernel argument defined as in instance of
            numba.types.
            val: The value of the kernel argument.

        Raises:
            UnsupportedKernelArgumentError: When the argument is of an
            unsupported type.

        """

        if isinstance(ty, USMNdArrayType):
            return self._unpack_usm_array(val)
        elif isinstance(ty, Array):
            return self._unpack_array(val)
        elif ty == types.int64:
            return ctypes.c_longlong(val)
        elif ty == types.uint64:
            return ctypes.c_ulonglong(val)
        elif ty == types.int32:
            return ctypes.c_int(val)
        elif ty == types.uint32:
            return ctypes.c_uint(val)
        elif ty == types.float64:
            return ctypes.c_double(val)
        elif ty == types.float32:
            return ctypes.c_float(val)
        elif ty == types.boolean:
            return ctypes.c_uint8(int(val))
        elif ty == types.complex64:
            raise UnsupportedKernelArgumentError(ty, val)
        elif ty == types.complex128:
            raise UnsupportedKernelArgumentError(ty, val)
        else:
            raise UnsupportedKernelArgumentError(ty, val)

    def _pack_array(self):
        """
        Copy device data back to host
        """
        for obj in self._repack_map.keys():

            (usm_mem, packed_ndarr, packed) = self._repack_map[obj]
            utils.copy_to_numpy_from_usm_obj(usm_mem, packed_ndarr)
            if packed:
                np.copyto(obj, packed_ndarr)

    def __init__(self, arg_list, argty_list, queue) -> None:
        """_summary_

        Args:
            arg_list (_type_): _description_
            argty_list (_type_): _description_
            queue: _description_
        """
        self._arg_list = arg_list
        self._argty_list = argty_list
        self._queue = queue

        # loop over the arg_list and generate the kernelargs list
        self._unpacked_args = []
        for i, val in enumerate(arg_list):
            arg = self._unpack_argument(ty=argty_list[i], val=val)
            if type(arg) == list:
                self._unpacked_args.extend(arg)
            else:
                self._unpacked_args.append(arg)

        # Create a map for numpy arrays storing the unpacked information, as
        # these arrays will need to be repacked.
        self._repack_map = {}

    @property
    def unpacked_args(self):
        return self._unpacked_args

    @property
    def repacked_args(self):
        self._pack_array()
        return self._repack_map.keys()
