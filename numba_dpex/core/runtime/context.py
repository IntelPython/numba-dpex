# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import functools

import numba.core.unsafe.nrt
from llvmlite import ir as llvmir
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
            msg = "USM allocation failed. Check the usm_type and queue."
            cgutils.guard_memory_error(self._context, builder, memptr, msg=msg)
            return memptr

        return wrap

    @_check_null_result
    def meminfo_alloc(self, builder, size, usm_type, queue_ref):
        """
        Wrapper to call :func:`~context.DpexRTContext.meminfo_alloc_unchecked`
        with null checking of the returned value.
        """

        return self.meminfo_alloc_unchecked(builder, size, usm_type, queue_ref)

    @_check_null_result
    def meminfo_fill(
        self,
        builder,
        meminfo,
        itemsize,
        dest_is_float,
        value_is_float,
        value,
        queue_ref,
    ):
        """
        Wrapper to call :func:`~context.DpexRTContext.meminfo_fill_unchecked`
        with null checking of the returned value.
        """
        return self.meminfo_fill_unchecked(
            builder,
            meminfo,
            itemsize,
            dest_is_float,
            value_is_float,
            value,
            queue_ref,
        )

    def meminfo_alloc_unchecked(self, builder, size, usm_type, queue_ref):
        """Allocate a new MemInfo with a data payload of `size` bytes.

        The result of the call is checked and if it is NULL, i.e. allocation
        failed, then a MemoryError is raised. If the allocation succeeded then
        a pointer to the MemInfo is returned.

        Args:
            builder (`llvmlite.ir.builder.IRBuilder`): LLVM IR builder.
            size (`llvmlite.ir.values.Argument`): LLVM uint64 value specifying
                the size in bytes for the data payload, i.e. i64 %"arg.allocsize"
            usm_type (`llvmlite.ir.values.Argument`): An LLVM Argument object
                specifying the type of the usm allocator. The constant value
                should match the values in
                ``dpctl's`` ``libsyclinterface::DPCTLSyclUSMType`` enum,
                i.e. i64 %"arg.usm_type".
            queue_ref (`llvmlite.ir.values.Argument`): An LLVM argument value storing
                the pointer to the address of the queue object, the object can be
                `dpctl.SyclQueue()`, i.e. i8* %"arg.queue".

        Returns:
            ret (`llvmlite.ir.instructions.CallInstr`): A pointer to the `MemInfo`
                is returned from the `DPEXRT_MemInfo_alloc` C function call.
        """

        mod = builder.module
        u64 = llvmir.IntType(64)
        fnty = llvmir.FunctionType(
            cgutils.voidptr_t,
            [cgutils.voidptr_t, cgutils.intp_t, u64, cgutils.voidptr_t],
        )
        fn = cgutils.get_or_insert_function(mod, fnty, "DPEXRT_MemInfo_alloc")
        fn.return_value.add_attribute("noalias")
        nrt_api = self._context.nrt.get_nrt_api(builder)

        ret = builder.call(fn, [nrt_api, size, usm_type, queue_ref])

        return ret

    def meminfo_fill_unchecked(
        self,
        builder,
        meminfo,
        itemsize,
        dest_is_float,
        value_is_float,
        value,
        queue_ref,
    ):
        """Fills an allocated `MemInfo` with the value specified.

        The result of the call is checked and if it is `NULL`, i.e. the fill
        operation failed, then a `MemoryError` is raised. If the fill operation
        is succeeded then a pointer to the `MemInfo` is returned.

        Args:
            builder (`llvmlite.ir.builder.IRBuilder`): LLVM IR builder.
            meminfo (`llvmlite.ir.instructions.LoadInstr`): LLVM uint64 value
                specifying the size in bytes for the data payload.
            itemsize (`llvmlite.ir.values.Constant`): An LLVM Constant value
                specifying the size of the each data item allocated by the
                usm allocator.
            dest_is_float (`llvmlite.ir.values.Constant`): An LLVM Constant
                value specifying if the destination array type is floating
                point.
            value_is_float (`llvmlite.ir.values.Constant`): An LLVM Constant
                value specifying if the input value is a floating point.
            value (`llvmlite.ir.values.Constant`): An LLVM Constant value
                specifying if the input value that will be used to fill
                the array.
            queue_ref (`llvmlite.ir.instructions.ExtractValue`): An LLVM ExtractValue
                instruction object to extract the pointer to the queue from the
                DpctlSyclQueue type, i.e. %".74" = extractvalue {i8*, i8*} %".73", 1.

        Returns:
            ret (`llvmlite.ir.instructions.CallInstr`): A pointer to the `MemInfo`
                is returned from the `DPEXRT_MemInfo_fill` C function call.
        """

        mod = builder.module
        u64 = llvmir.IntType(64)
        b = llvmir.IntType(1)
        fnty = llvmir.FunctionType(
            cgutils.voidptr_t,
            [cgutils.voidptr_t, u64, b, b, u64, cgutils.voidptr_t],
        )
        fn = cgutils.get_or_insert_function(mod, fnty, "DPEXRT_MemInfo_fill")
        fn.return_value.add_attribute("noalias")

        ret = builder.call(
            fn,
            [
                meminfo,
                itemsize,
                dest_is_float,
                value_is_float,
                value,
                queue_ref,
            ],
        )

        return ret

    def arraystruct_from_python(self, pyapi, obj, ptr):
        """Generates a call to DPEXRT_sycl_usm_ndarray_from_python C function
        defined in the _DPREXRT_python Python extension.

        """
        fnty = llvmir.FunctionType(
            llvmir.IntType(32), [pyapi.voidptr, pyapi.pyobj, pyapi.voidptr]
        )
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)
        fn = pyapi._get_function(fnty, "DPEXRT_sycl_usm_ndarray_from_python")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
        fn.args[2].add_attribute("nocapture")

        self.error = pyapi.builder.call(fn, (nrt_api, obj, ptr))

        return self.error

    def queuestruct_from_python(self, pyapi, obj, ptr):
        """Calls the c function DPEXRT_sycl_queue_from_python"""
        fnty = llvmir.FunctionType(
            llvmir.IntType(32), [pyapi.voidptr, pyapi.pyobj, pyapi.voidptr]
        )
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)

        fn = pyapi._get_function(fnty, "DPEXRT_sycl_queue_from_python")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
        fn.args[2].add_attribute("nocapture")

        self.error = pyapi.builder.call(fn, (nrt_api, obj, ptr))
        return self.error

    def queuestruct_to_python(self, pyapi, val):
        """Calls the c function DPEXRT_sycl_queue_to_python"""

        fnty = llvmir.FunctionType(pyapi.pyobj, [pyapi.voidptr, pyapi.voidptr])
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)

        fn = pyapi._get_function(fnty, "DPEXRT_sycl_queue_to_python")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")

        qptr = cgutils.alloca_once_value(pyapi.builder, val)
        ptr = pyapi.builder.bitcast(qptr, pyapi.voidptr)

        self.error = pyapi.builder.call(fn, [nrt_api, ptr])

        return self.error

    def eventstruct_from_python(self, pyapi, obj, ptr):
        """Calls the c function DPEXRT_sycl_event_from_python"""
        fnty = llvmir.FunctionType(
            llvmir.IntType(32), [pyapi.voidptr, pyapi.pyobj, pyapi.voidptr]
        )
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)

        fn = pyapi._get_function(fnty, "DPEXRT_sycl_event_from_python")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
        fn.args[2].add_attribute("nocapture")

        self.error = pyapi.builder.call(fn, (nrt_api, obj, ptr))
        return self.error

    def eventstruct_to_python(self, pyapi, val):
        """Calls the c function DPEXRT_sycl_event_to_python"""

        fnty = llvmir.FunctionType(pyapi.pyobj, [pyapi.voidptr, pyapi.voidptr])
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)

        fn = pyapi._get_function(fnty, "DPEXRT_sycl_event_to_python")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")

        qptr = cgutils.alloca_once_value(pyapi.builder, val)
        ptr = pyapi.builder.bitcast(qptr, pyapi.voidptr)

        self.error = pyapi.builder.call(fn, [nrt_api, ptr])

        return self.error

    def eventstruct_init(self, pyapi, event, struct):
        """Calls the c function DPEXRT_sycl_event_init"""

        fnty = llvmir.FunctionType(
            llvmir.IntType(32), [pyapi.voidptr, pyapi.voidptr, pyapi.voidptr]
        )
        nrt_api = self._context.nrt.get_nrt_api(pyapi.builder)

        fn = pyapi._get_function(fnty, "DPEXRT_sycl_event_init")
        fn.args[0].add_attribute("nocapture")
        fn.args[1].add_attribute("nocapture")
        fn.args[2].add_attribute("nocapture")

        ptr = pyapi.builder.bitcast(struct, pyapi.voidptr)
        self.error = pyapi.builder.call(fn, [nrt_api, event, ptr])

        return self.error

    def usm_ndarray_to_python_acqref(self, pyapi, aryty, ary, dtypeptr):
        """Boxes a DpnpNdArray native object into a Python dpnp.ndarray.

        Args:
            pyapi (_type_): _description_
            aryty (_type_): _description_
            ary (_type_): _description_
            dtypeptr (_type_): _description_

        Returns:
            _type_: _description_
        """
        argtys = [
            pyapi.voidptr,
            pyapi.pyobj,
            llvmir.IntType(32),
            llvmir.IntType(32),
            pyapi.pyobj,
        ]
        fnty = llvmir.FunctionType(pyapi.pyobj, argtys)
        fn = pyapi._get_function(
            fnty, "DPEXRT_sycl_usm_ndarray_to_python_acqref"
        )
        fn.args[0].add_attribute("nocapture")

        aryptr = cgutils.alloca_once_value(pyapi.builder, ary)
        ptr = pyapi.builder.bitcast(aryptr, pyapi.voidptr)

        # Embed the Python type of the array (maybe subclass) in the LLVM IR.
        serialized = pyapi.serialize_object(aryty.box_type)
        serial_aryty_pytype = pyapi.unserialize(serialized)

        ndim = pyapi.context.get_constant(types.int32, aryty.ndim)
        writable = pyapi.context.get_constant(types.int32, int(aryty.mutable))

        args = [ptr, serial_aryty_pytype, ndim, writable, dtypeptr]
        return pyapi.builder.call(fn, args)

    def get_queue_from_filter_string(self, builder, device):
        """Calls DPEXRTQueue_CreateFromFilterString to create a new sycl::queue
        from a given filter string.

        Args:
            device (llvmlite.ir.values.FormattedConstant): An LLVM ArrayType
                storing a const string for a DPC++ filter selector string.

        Returns: A DPCTLSyclQueueRef pointer.
        """
        mod = builder.module
        fnty = llvmir.FunctionType(
            cgutils.voidptr_t,
            [cgutils.voidptr_t],
        )
        fn = cgutils.get_or_insert_function(
            mod, fnty, "DPEXRTQueue_CreateFromFilterString"
        )
        fn.return_value.add_attribute("noalias")

        ret = builder.call(fn, [device])

        return ret

    def submit_range(
        self,
        builder,
        kref,
        qref,
        args,
        argtys,
        nargs,
        range,
        nrange,
        depevents,
        ndepevents,
    ):
        """Calls DPEXRTQueue_CreateFromFilterString to create a new sycl::queue
        from a given filter string.

        Returns: A DPCTLSyclQueueRef pointer.
        """
        mod = builder.module
        fnty = llvmir.FunctionType(
            llvmir.types.VoidType(),
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t.as_pointer(),
                cgutils.int32_t.as_pointer(),
                llvmir.IntType(64),
                llvmir.IntType(64).as_pointer(),
                llvmir.IntType(64),
                cgutils.voidptr_t,
                llvmir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            mod, fnty, "DpexrtQueue_SubmitRange"
        )

        ret = builder.call(
            fn,
            [
                kref,
                qref,
                args,
                argtys,
                nargs,
                range,
                nrange,
                depevents,
                ndepevents,
            ],
        )

        return ret

    def submit_ndrange(
        self,
        builder,
        kref,
        qref,
        args,
        argtys,
        nargs,
        grange,
        lrange,
        ndims,
        depevents,
        ndepevents,
    ):
        """Calls DPEXRTQueue_CreateFromFilterString to create a new sycl::queue
        from a given filter string.

        Returns: A LLVM IR call inst.
        """

        mod = builder.module
        fnty = llvmir.FunctionType(
            llvmir.types.VoidType(),
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t.as_pointer(),
                cgutils.int32_t.as_pointer(),
                llvmir.IntType(64),
                llvmir.IntType(64).as_pointer(),
                llvmir.IntType(64).as_pointer(),
                llvmir.IntType(64),
                cgutils.voidptr_t,
                llvmir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            mod, fnty, "DpexrtQueue_SubmitNDRange"
        )

        ret = builder.call(
            fn,
            [
                kref,
                qref,
                args,
                argtys,
                nargs,
                grange,
                lrange,
                ndims,
                depevents,
                ndepevents,
            ],
        )

        return ret

    def acquire_meminfo_and_schedule_release(
        self, builder: llvmir.IRBuilder, args
    ):
        """Inserts LLVM IR to call nrt_acquire_meminfo_and_schedule_release.

        .. code-block:: c

            DPCTLSyclEventRef
            DPEXRT_nrt_acquire_meminfo_and_schedule_release(
                NRT_api_functions *nrt,
                DPCTLSyclQueueRef QRef,
                NRT_MemInfo **meminfo_array,
                size_t meminfo_array_size,
                DPCTLSyclEventRef *depERefs,
                size_t nDepERefs,
                int *status,
            );

        """
        mod = builder.module

        func_ty = llvmir.FunctionType(
            cgutils.voidptr_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t.as_pointer(),
                llvmir.IntType(64),
                cgutils.voidptr_t.as_pointer(),
                llvmir.IntType(64),
                llvmir.IntType(64).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            mod, func_ty, "DPEXRT_nrt_acquire_meminfo_and_schedule_release"
        )
        ret = builder.call(fn, args)

        return ret

    def build_or_get_kernel(self, builder: llvmir.IRBuilder, args):
        """Inserts LLVM IR to call build_or_get_kernel.

        .. code-block:: c

            DPCTLSyclKernelRef
            DPEXRT_build_or_get_kernel(
                const DPCTLSyclContextRef ctx,
                const DPCTLSyclDeviceRef dev,
                size_t il_hash,
                const char *il,
                size_t il_length,
                const char *compile_opts,
                const char *kernel_name,
            );

        """
        mod = builder.module

        func_ty = llvmir.FunctionType(
            cgutils.voidptr_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                llvmir.IntType(64),
                cgutils.voidptr_t,
                llvmir.IntType(64),
                cgutils.voidptr_t,
                cgutils.voidptr_t,
            ],
        )
        fn = cgutils.get_or_insert_function(
            mod, func_ty, "DPEXRT_build_or_get_kernel"
        )
        ret = builder.call(fn, args)

        return ret

    def kernel_cache_size(self, builder: llvmir.IRBuilder):
        """Inserts LLVM IR to call kernel_cache_size.

        .. code-block:: c

            size_t DPEXRT_kernel_cache_size();

        """
        fn = cgutils.get_or_insert_function(
            builder.module,
            llvmir.FunctionType(
                llvmir.IntType(64),
                [],
            ),
            "DPEXRT_kernel_cache_size",
        )

        return builder.call(fn, [])
