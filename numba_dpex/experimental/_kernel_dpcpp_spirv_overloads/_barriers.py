# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides overloads for functions included in kernel_iface.barrier that
generate dpcpp SPIR-V LLVM IR intrinsic function calls.
"""
from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.experimental.kernel_iface import (
    group_barrier,
    sub_group_barrier,
)
from numba_dpex.experimental.kernel_iface.memory_enums import (
    MemoryOrder,
    MemoryScope,
)
from numba_dpex.experimental.target import DPEX_KERNEL_EXP_TARGET_NAME

from ._spv_atomic_inst_helper import get_memory_semantics_mask, get_scope


def get_memory_scope(fence_scope):
    if isinstance(fence_scope, types.Literal):
        return get_scope(fence_scope.literal_value)
    else:
        return get_scope(fence_scope.value)


@intrinsic
def _intrinsic_barrier(
    ty_context, ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask
):
    sig = types.void(ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask)

    def _intrinsic_barrier_codegen(context, builder, sig, args):
        fn_name = "__spirv_ControlBarrier"
        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            fn_name, [ty_exec_scope, ty_mem_scope, ty_spirv_mem_sem_mask]
        )

        spirv_fn_arg_types = [context.get_value_type(t) for t in sig.args]

        fnty = llvmir.FunctionType(llvmir.VoidType(), spirv_fn_arg_types)

        fn_args = [args[0], args[1], args[2]]

        fn = cgutils.get_or_insert_function(
            builder.module, fnty, mangled_fn_name
        )
        # XXX Uncomment once llvmlite PR#1019 is merged and available for use
        # fn.attributes.add("convergent")
        # fn.attributes.add("nounwind")
        fn.calling_convention = "spir_func"

        builder.call(fn, fn_args)

        # XXX Uncomment once llvmlite PR#1019 is merged and available for use
        # callinst.attributes.add("convergent")
        # callinst.attributes.add("nounwind")

        return

    return (
        sig,
        _intrinsic_barrier_codegen,
    )


@overload(
    group_barrier,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def _ol_group_barrier(fence_scope=MemoryScope.WORK_GROUP):
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.SEQ_CST.value
    )
    exec_scope = get_scope(MemoryScope.WORK_GROUP.value)
    mem_scope = get_memory_scope(fence_scope)

    def _ol_group_barrier_impl(fence_scope=MemoryScope.WORK_GROUP):
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return _ol_group_barrier_impl


@overload(
    sub_group_barrier,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def _ol_sub_group_barrier(fence_scope=MemoryScope.SUB_GROUP):
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.SEQ_CST.value
    )
    exec_scope = get_scope(MemoryScope.SUB_GROUP.value)
    mem_scope = get_memory_scope(fence_scope)

    def _ol_sub_group_barrier_impl(fence_scope=MemoryScope.SUB_GROUP):
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return _ol_sub_group_barrier_impl
