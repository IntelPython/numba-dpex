# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""llvmlite function call generators for dpctl's libsyclinterface library"""

from llvmlite import ir as llvmir
from numba.core import cgutils


def _build_dpctl_function(llvm_module, return_ty, arg_list, func_name):
    """
    _build_dpctl_function(llvm_module, return_ty, arg_list, func_name)
    Inserts an LLVM function of the specified signature and name into an
    LLVM module.

    Args:
        llvm_module: The LLVM IR Module into which the function will be inserted
        return_ty: An LLVM Value corresponding to the return type of the
                   function.
        arg_list: A list of LLVM Value objects corresponding to the
                  type of the arguments for the function.
        func_name: The name of the function passed as a string.

    Return: A Python object wrapping an LLVM Function.

    """
    func_ty = llvmir.FunctionType(return_ty, arg_list)
    fn = cgutils.get_or_insert_function(llvm_module, func_ty, func_name)
    return fn


# --------------------- DpctlSyclQueue functions ------------------------------#


def dpctl_queue_copy(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_Copy to create a copy of a
    DpctlSyclQueueRef pointer passed in to the function.
    """
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLQueue_Copy",
    )

    ret = builder.call(fn, args)

    return ret


def dpctl_queue_memcpy(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_Memcpy."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            cgutils.intp_t,
        ],
        func_name="DPCTLQueue_Memcpy",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_event_create(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLEvent_Create."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[],
        func_name="DPCTLEvent_Create",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_queue_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.types.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLQueue_Delete",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_queue_get_context(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_GetContext."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLQueue_GetContext",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_queue_get_device(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_GetDevice."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLQueue_GetDevice",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_queue_submit_range(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_SubmitRange.

    DPCTLSyclEventRef
    DPCTLQueue_SubmitRange(
        const DPCTLSyclKernelRef KRef,
        const DPCTLSyclQueueRef QRef,
        void** Args,
        const DPCTLKernelArgType* ArgTypes,
        size_t NArgs,
        const size_t Range[3],
        size_t NRange,
        const DPCTLSyclEventRef* DepEvents,
        size_t NDepEvents
    );

    """
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            cgutils.voidptr_t.as_pointer(),
            cgutils.int32_t.as_pointer(),
            llvmir.IntType(64),
            llvmir.IntType(64).as_pointer(),
            llvmir.IntType(64),
            cgutils.voidptr_t.as_pointer(),
            llvmir.IntType(64),
        ],
        func_name="DPCTLQueue_SubmitRange",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_queue_submit_ndrange(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_SubmitNDRange.

    DPCTLSyclEventRef
    DPCTLQueue_SubmitNDRange(
        const DPCTLSyclKernelRef KRef,
        const DPCTLSyclQueueRef QRef,
        void** Args,
        const DPCTLKernelArgType* ArgTypes,
        size_t NArgs,
        const size_t gRange[3],
        const size_t lRange[3],
        size_t NDims,
        const DPCTLSyclEventRef* DepEvents,
        size_t NDepEvents
    );

    """
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            cgutils.voidptr_t.as_pointer(),
            cgutils.int32_t.as_pointer(),
            llvmir.IntType(64),
            llvmir.IntType(64).as_pointer(),
            llvmir.IntType(64).as_pointer(),
            llvmir.IntType(64),
            cgutils.voidptr_t.as_pointer(),
            llvmir.IntType(64),
        ],
        func_name="DPCTLQueue_SubmitNDRange",
    )
    ret = builder.call(fn, args)

    return ret


# --------------------- DpctlSyclDevice functions -----------------------------#


def dpctl_device_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLDevice_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.types.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLDevice_Delete",
    )
    ret = builder.call(fn, args)

    return ret


# --------------------- DpctlSyclKernelBundle functions -----------------------#


def dpctl_kernel_bundle_create_from_spirv(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLKernelBundle_CreateFromSpirv."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            cgutils.voidptr_t,
            llvmir.IntType(64),
            cgutils.voidptr_t,
        ],
        func_name="DPCTLKernelBundle_CreateFromSpirv",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_kernel_bundle_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLKernelBundle_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.types.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLKernelBundle_Delete",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_kernel_bundle_get_kernel(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLKernelBundle_GetKernel."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=cgutils.voidptr_t,
        arg_list=[cgutils.voidptr_t, cgutils.voidptr_t],
        func_name="DPCTLKernelBundle_GetKernel",
    )
    ret = builder.call(fn, args)

    return ret


# --------------------- DpctlSyclKernel functions -----------------------------#


def dpctl_kernel_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLKernel_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.types.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLKernel_Delete",
    )
    ret = builder.call(fn, args)

    return ret


# --------------------- DpctlSyclContext functions ----------------------------#


def dpctl_context_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLContext_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.types.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLContext_Delete",
    )
    ret = builder.call(fn, args)

    return ret


# --------------------- DpctlSyclEvent functions ------------------------------#


def dpctl_event_wait(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLEvent_Wait."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLEvent_Wait",
    )
    ret = builder.call(fn, args)

    return ret


def dpctl_event_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLEvent_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLEvent_Delete",
    )
    ret = builder.call(fn, args)

    return ret
