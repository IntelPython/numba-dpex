# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
import logging

import dpctl.memory as dpctl_mem
import numpy as np
from numba.core import types

import numba_dpex.utils as utils
from numba_dpex.core.exceptions import UnsupportedKernelArgumentError
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
    argument to a numba_dpex kernel function into corresponding ctype object.
    """

    def _unpack_usm_array(self, val):
        """Flattens an object of USMNdArray type into ctypes objects to be
        passed as kernel arguments.

        Args:
            val : An object of dpctl.types.UsmNdArray type.

        Returns:
            list: A list of ctype objects representing the flattened usm_ndarray
        """
        unpacked_array_attrs = []
        suai_attrs = get_info_from_suai(val)
        size = suai_attrs.size
        itemsize = suai_attrs.itemsize
        buf = suai_attrs.data
        shape = suai_attrs.shape
        strides = suai_attrs.strides
        ndim = suai_attrs.dimensions

        unpacked_array_attrs.append(ctypes.c_longlong(size))
        unpacked_array_attrs.append(ctypes.c_longlong(itemsize))
        unpacked_array_attrs.append(buf)
        for ax in range(ndim):
            unpacked_array_attrs.append(ctypes.c_longlong(shape[ax]))
        for ax in range(ndim):
            unpacked_array_attrs.append(ctypes.c_longlong(strides[ax]))

        return unpacked_array_attrs

    def _unpack_argument(self, ty, val):
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
            return [ctypes.c_float(val.real), ctypes.c_float(val.imag)]
        elif ty == types.complex128:
            return [ctypes.c_double(val.real), ctypes.c_double(val.imag)]
        else:
            raise UnsupportedKernelArgumentError(ty, val, self._pyfunc_name)

    def __init__(self, kernel_name, arg_list, argty_list, queue) -> None:
        """Initializes new Packer object and unpacks the input argument list.

        Args:
            kernel_name (str): The kernel function name.
            arg_list (list): A list of arguments to be unpacked
            argty_list (list): A list of Numba inferred types for each argument.
        """
        self._pyfunc_name = kernel_name
        self._arg_list = arg_list
        self._argty_list = argty_list

        # loop over the arg_list and generate the kernelargs list
        self._unpacked_args = []
        for i, val in enumerate(arg_list):
            arg = self._unpack_argument(ty=argty_list[i], val=val)
            if type(arg) == list:
                self._unpacked_args.extend(arg)
            else:
                self._unpacked_args.append(arg)

    @property
    def unpacked_args(self):
        """Returns the list of unpacked arguments created by a Packer object."""
        return self._unpacked_args
