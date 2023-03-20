# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging

import dpctl.memory as dpctl_mem
import numpy as np
from numba.core import types

import numba_dpex.utils as utils
from numba_dpex.core.exceptions import (
    UnsupportedAccessQualifierError,
    UnsupportedKernelArgumentError,
)
from numba_dpex.core.types import USMNdArray
from numba_dpex.core.utils import get_info_from_suai


class _NumPyArrayPackerPayload:
    def __init__(self, usm_mem, orig_val, packed_val, packed) -> None:
        self._usm_mem = usm_mem
        self._orig_val = orig_val
        self._packed_val = packed_val
        self._packed = packed


class Packer:
    """Implements the functionality to unpack a Python object passed as an
    argument to a numba_dpex kernel fucntion into corresponding ctype object.
    """

    # TODO: Remove after NumPy support is removed
    _access_types = ("read_only", "write_only", "read_write")

    def _check_for_invalid_access_type(self, array_val, access_type):
        if access_type and access_type not in Packer._access_types:
            raise UnsupportedAccessQualifierError(
                self._pyfunc_name,
                array_val,
                access_type,
                ",".join(Packer._access_types),
            )

    def _unpack_array_helper(self, size, itemsize, buf, shape, strides, ndim):
        """
        Implements the unpacking logic for array arguments.

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

        # meminfo (FIXME: should be removed and the USMNdArray type modified
        # once NumPy support is removed)
        unpacked_array_attrs.append(ctypes.c_size_t(0))
        # parent (FIXME: Evaluate if the attribute should be removed and the
        # USMNdArray type modified once NumPy support is removed)
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
        """Flattens an object of USMNdArray type into ctypes objects to be
        passed as kernel arguments.

        Args:
            val : An object of dpctl.types.UsmNdArray type.

        Returns:
            list: A list of ctype objects representing the flattened usm_ndarray
        """
        suai_attrs = get_info_from_suai(val)

        return self._unpack_array_helper(
            size=suai_attrs.size,
            itemsize=suai_attrs.itemsize,
            buf=suai_attrs.data,
            shape=suai_attrs.shape,
            strides=suai_attrs.strides,
            ndim=suai_attrs.dimensions,
        )

    def _unpack_array(self, val, access_type):
        """Deprecated to be removed once NumPy array support in kernels is
        removed.
        """
        packed_val = val
        # Check if the NumPy array is backed by USM memory
        usm_mem = utils.has_usm_memory(val)

        # If the NumPy array is not USM backed, then copy to a USM memory
        # object. Add an entry to the repack_map so that on exit from kernel
        # the data from the USM object can be copied back into the NumPy array.
        if usm_mem is None:
            self._check_for_invalid_access_type(val, access_type)
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
                self._repack_list.append(
                    _NumPyArrayPackerPayload(
                        usm_mem, orig_val, packed_val, packed
                    )
                )
            elif access_type == "write_only":
                self._repack_list.append(
                    _NumPyArrayPackerPayload(
                        usm_mem, orig_val, packed_val, packed
                    )
                )
            else:
                utils.copy_from_numpy_to_usm_obj(usm_mem, packed_val)
                self._repack_list.append(
                    _NumPyArrayPackerPayload(
                        usm_mem, orig_val, packed_val, packed
                    )
                )

        return self._unpack_array_helper(
            packed_val.size,
            packed_val.dtype.itemsize,
            usm_mem,
            packed_val.shape,
            packed_val.strides,
            packed_val.ndim,
        )

    def _unpack_argument(self, ty, val, access_specifier):
        """
        Unpack a Python object into one or more ctype values using Numba's
        type-inference machinery.

        Args:
            ty: The data types of the kernel argument defined as in instance of
            numba.types.
            val: The value of the kernel argument.

        Raises:
            UnsupportedKernelArgumentError: When the argument is of an
            unsupported type.

        """

        if isinstance(ty, USMNdArray):
            return self._unpack_usm_array(val)
        elif isinstance(ty, types.Array):
            return self._unpack_array(val, access_specifier)
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
            raise UnsupportedKernelArgumentError(ty, val, self._pyfunc_name)
        elif ty == types.complex128:
            raise UnsupportedKernelArgumentError(ty, val, self._pyfunc_name)
        else:
            raise UnsupportedKernelArgumentError(ty, val, self._pyfunc_name)

    def _pack_array(self):
        """
        Deprecated to be removed once NumPy array support in kernels is
        removed.
        """
        for obj in self._repack_list:
            utils.copy_to_numpy_from_usm_obj(obj._usm_mem, obj._packed_val)
            if obj._packed:
                np.copyto(obj._orig_val, obj._packed_val)

    def __init__(
        self, kernel_name, arg_list, argty_list, access_specifiers_list, queue
    ) -> None:
        """Initializes new Packer object and unpacks the input argument list.

        Args:
            arg_list (list): A list of arguments to be unpacked
            argty_list (list): A list of Numba inferred types for each argument.
            access_specifiers_list(list): A list of access specifiers for
            NumPy arrays to optimize host to device memory copy.
            [Deprecated: can be removed along with NumPy array support]
            queue (dpctl.SyclQueue): The SYCL queue where the kernel is to be
            executed. The queue is required to allocate USM memory for NumPy
            arrays.
            [Deprecated: can be removed along with NumPy array support]
        """
        self._pyfunc_name = kernel_name
        self._arg_list = arg_list
        self._argty_list = argty_list
        self._queue = queue
        # Create a list to store the numpy arrays that need to be
        # repacked beoe returning from a kernel.
        self._repack_list = []

        # loop over the arg_list and generate the kernelargs list
        self._unpacked_args = []
        for i, val in enumerate(arg_list):
            arg = self._unpack_argument(
                ty=argty_list[i],
                val=val,
                access_specifier=access_specifiers_list[i],
            )
            if type(arg) == list:
                self._unpacked_args.extend(arg)
            else:
                self._unpacked_args.append(arg)

    @property
    def unpacked_args(self):
        """Returns the list of unpacked arguments created by a Packer object."""
        return self._unpacked_args

    @property
    def repacked_args(self):
        """Returns the list of NumPy"""
        self._pack_array()
        return self._repack_list
