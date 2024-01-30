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


def _get_memory_scope(fence_scope):
    if isinstance(fence_scope, types.Literal):
        return get_scope(fence_scope.literal_value)
    return get_scope(fence_scope.value)


@intrinsic
def _intrinsic_barrier(
    ty_context,  # pylint: disable=unused-argument
    ty_exec_scope,  # pylint: disable=unused-argument
    ty_mem_scope,  # pylint: disable=unused-argument
    ty_spirv_mem_sem_mask,  # pylint: disable=unused-argument
):
    # Signature of `__spirv_control_barrier` call that is
    # generated for group_barrier. It takes three arguments -
    # exec_scope, memory_scope and memory_semantics_mask.
    # All arguments have to be of type unsigned int32.
    sig = types.void(types.uint32, types.uint32, types.uint32)

    def _intrinsic_barrier_codegen(
        context, builder, sig, args  # pylint: disable=unused-argument
    ):
        exec_scope_arg = builder.trunc(args[0], llvmir.IntType(32))
        mem_scope_arg = builder.trunc(args[1], llvmir.IntType(32))
        spirv_memory_semantics_mask_arg = builder.trunc(
            args[2], llvmir.IntType(32)
        )

        fn_args = [
            exec_scope_arg,
            mem_scope_arg,
            spirv_memory_semantics_mask_arg,
        ]

        mangled_fn_name = ext_itanium_mangler.mangle_ext(
            "__spirv_ControlBarrier", [types.uint32, types.uint32, types.uint32]
        )

        spirv_fn_arg_types = [
            llvmir.IntType(32),
            llvmir.IntType(32),
            llvmir.IntType(32),
        ]

        fn = cgutils.get_or_insert_function(
            builder.module,
            llvmir.FunctionType(llvmir.VoidType(), spirv_fn_arg_types),
            mangled_fn_name,
        )

        fn.attributes.add("convergent")
        fn.attributes.add("nounwind")
        fn.calling_convention = "spir_func"

        callinst = builder.call(fn, fn_args)

        callinst.attributes.add("convergent")
        callinst.attributes.add("nounwind")

    return (
        sig,
        _intrinsic_barrier_codegen,
    )


@overload(
    group_barrier,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def ol_group_barrier(fence_scope=MemoryScope.WORK_GROUP):
    """SPIR-V overload for
    :meth:`numba_dpex.experimental.kernel_iface.group_barrier`.

    Generates the same LLVM IR instruction as dpcpp for the
    `group_barrier` function.
    """

    # Per SYCL spec, group_barrier must perform both control
    # barrier and memory fence operations. Hence,
    # group_barrier requires two scopes and memory
    # consistency specification as three arguments.
    #
    # mem_scope - scope of any memory consistency operations
    #             that are performed by the barrier. By default,
    #             mem_scope is set to `work_group`.
    # exec_scope - scope that determines the set of work-items
    #              that synchronize at barrier.
    #              Set to `work_group` for group_barrier always.
    # spirv_memory_semantics_mask - Based on sycl implementation,
    # Mask that is set to use sequential consistency
    # memory order semantics always.

    mem_scope = _get_memory_scope(fence_scope)
    exec_scope = get_scope(MemoryScope.WORK_GROUP.value)
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.SEQ_CST.value
    )

    def _ol_group_barrier_impl(
        fence_scope=MemoryScope.WORK_GROUP,
    ):  # pylint: disable=unused-argument
        # pylint: disable=no-value-for-parameter
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return _ol_group_barrier_impl


@overload(
    sub_group_barrier,
    prefer_literal=True,
    target=DPEX_KERNEL_EXP_TARGET_NAME,
)
def ol_sub_group_barrier(fence_scope=MemoryScope.SUB_GROUP):
    """SPIR-V overload for
    :meth:`numba_dpex.experimental.kernel_iface.sub_group_barrier`.

    Generates the same LLVM IR instruction as dpcpp for the
    `sub_group_barrier` function.
    """
    mem_scope = _get_memory_scope(fence_scope)
    # exec_scope is always set to `sub_group`
    exec_scope = get_scope(MemoryScope.SUB_GROUP.value)
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.SEQ_CST.value
    )

    def _ol_sub_group_barrier_impl(
        fence_scope=MemoryScope.SUB_GROUP,
    ):  # pylint: disable=unused-argument
        # pylint: disable=no-value-for-parameter
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return _ol_sub_group_barrier_impl
