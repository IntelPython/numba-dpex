# SPDX-FileCopyrightText: 2023 Intel Corporation
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


def dpctl_queue_delete(builder: llvmir.IRBuilder, *args):
    """Inserts LLVM IR to call DPCTLQueue_Delete."""
    mod = builder.module
    fn = _build_dpctl_function(
        llvm_module=mod,
        return_ty=llvmir.VoidType(),
        arg_list=[cgutils.voidptr_t],
        func_name="DPCTLQueue_Delete",
    )
    ret = builder.call(fn, args)

    return ret
