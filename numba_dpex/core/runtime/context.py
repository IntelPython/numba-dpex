# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import functools

import llvmlite.llvmpy.core as lc
from llvmlite import ir
from llvmlite.llvmpy.core import ATTR_NO_CAPTURE, Type
from numba.core import cgutils, types


class DpexRTContext(object):
    """
    An object providing access to DPEXRT API in the lowering pass.
    """

    def __init__(self, context):
        self._context = context

    def _check_null_result(func):
        @functools.wraps(func)
        def wrap(self, builder, *args, **kwargs):
            memptr = func(self, builder, *args, **kwargs)
            msg = "USM allocation failed. Check the usm_type and filter "
            "string values."
            cgutils.guard_memory_error(self._context, builder, memptr, msg=msg)
            return memptr

        return wrap

    @_check_null_result
    def meminfo_alloc(self, builder, size, usm_type, device):
        """Allocate a new MemInfo with a data payload of `size` bytes.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised. If the allocation succeeded then
        a pointer to the MemInfo is returned.

        Args:
            builder (_type_): LLVM IR builder
            size (_type_): LLVM uint64 Value specifying the size in bytes for
            the data payload.
            usm_type (_type_): An LLVM Constant Value specifying the type of the
            usm allocator. The constant value should match the values in
            ``dpctl's`` ``libsyclinterface::DPCTLSyclUSMType`` enum.
            device (_type_): An LLVM ArrayType storing a const string for a
            DPC++ filter selector string.

        Returns: A pointer to the MemInfo is returned.
        """

        return self.meminfo_alloc_unchecked(builder, size, usm_type, device)

    def meminfo_alloc_unchecked(self, builder, size, usm_type, device):
        """
        Allocate a new MemInfo with a data payload of `size` bytes.

        A pointer to the MemInfo is returned.

        Returns NULL to indicate error/failure to allocate.
        """
        mod = builder.module
        u64 = ir.IntType(64)
        fnty = ir.FunctionType(
            cgutils.voidptr_t, [cgutils.intp_t, u64, cgutils.voidptr_t]
        )
        fn = cgutils.get_or_insert_function(mod, fnty, "DPEXRT_MemInfo_alloc")
        fn.return_value.add_attribute("noalias")

        ret = builder.call(fn, [size, usm_type, device])

        return ret

    def arraystruct_from_python(self, pyapi, obj, ptr):
        """Generates a call to DPEXRT_sycl_usm_ndarray_from_python C function
        defined in the _DPREXRT_python Python extension.

        Args:
            pyapi (_type_): _description_
            obj (_type_): _description_
            ptr (_type_): _description_

        Returns:
            _type_: _description_
        """
        fnty = Type.function(Type.int(), [pyapi.pyobj, pyapi.voidptr])
        fn = pyapi._get_function(fnty, "DPEXRT_sycl_usm_ndarray_from_python")
        fn.args[0].add_attribute(lc.ATTR_NO_CAPTURE)
        fn.args[1].add_attribute(lc.ATTR_NO_CAPTURE)

        self.error = pyapi.builder.call(fn, (obj, ptr))

        return self.error

    def usm_ndarray_to_python_acqref(self, pyapi, aryty, ary, dtypeptr):
        """_summary_

        Args:
            pyapi (_type_): _description_
            aryty (_type_): _description_
            ary (_type_): _description_
            dtypeptr (_type_): _description_

        Returns:
            _type_: _description_
        """
        args = [
            pyapi.voidptr,
            pyapi.pyobj,
            ir.IntType(32),
            ir.IntType(32),
            pyapi.pyobj,
        ]
        fnty = Type.function(pyapi.pyobj, args)
        fn = pyapi._get_function(
            fnty, "DPEXRT_sycl_usm_ndarray_to_python_acqref"
        )
        fn.args[0].add_attribute(ATTR_NO_CAPTURE)

        aryptr = cgutils.alloca_once_value(pyapi.builder, ary)
        ptr = pyapi.builder.bitcast(aryptr, pyapi.voidptr)

        # Embed the Python type of the array (maybe subclass) in the LLVM IR.
        serialized = pyapi.serialize_object(aryty.box_type)
        serial_aryty_pytype = pyapi.unserialize(serialized)

        ndim = pyapi.context.get_constant(types.int32, aryty.ndim)
        writable = pyapi.context.get_constant(types.int32, int(aryty.mutable))

        args = [ptr, serial_aryty_pytype, ndim, writable, dtypeptr]
        return pyapi.builder.call(fn, args)
