# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides overloads for functions included in kernel_api.barrier that
generate dpcpp SPIR-V LLVM IR intrinsic function calls.
"""

from llvmlite import ir as llvmir
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload

from numba_dpex.core.types.kernel_api.index_space_ids import GroupType
from numba_dpex.kernel_api import group_barrier
from numba_dpex.kernel_api.memory_enums import MemoryOrder, MemoryScope
from numba_dpex.kernel_api_impl.spirv.target import SPIRV_TARGET_NAME

from ._spv_atomic_inst_helper import get_memory_semantics_mask, get_scope
from .spv_fn_declarations import (
    _SUPPORT_CONVERGENT,
    get_or_insert_spv_group_barrier_fn,
)


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
    # Signature of `__spirv_ControlBarrier` call that is
    # generated for group_barrier. It takes three arguments -
    # exec_scope, memory_scope and memory_semantics_mask.
    # All arguments have to be of type unsigned int32.
    sig = types.void(types.uint32, types.uint32, types.uint32)

    def _intrinsic_barrier_codegen(
        context, builder, sig, args  # pylint: disable=unused-argument
    ):
        fn_args = [
            builder.trunc(args[0], llvmir.IntType(32)),
            builder.trunc(args[1], llvmir.IntType(32)),
            builder.trunc(args[2], llvmir.IntType(32)),
        ]

        callinst = builder.call(
            get_or_insert_spv_group_barrier_fn(builder.module), fn_args
        )

        if _SUPPORT_CONVERGENT:  # pylint: disable=duplicate-code
            callinst.attributes.add("convergent")
        callinst.attributes.add("nounwind")

    return (
        sig,
        _intrinsic_barrier_codegen,
    )


@overload(
    group_barrier,
    prefer_literal=True,
    target=SPIRV_TARGET_NAME,
)
def ol_group_barrier(group, fence_scope=MemoryScope.WORK_GROUP):
    """SPIR-V overload for
    :meth:`numba_dpex.kernel_api.group_barrier`.

    Generates the same LLVM IR instruction as DPC++ for the SYCL
    `group_barrier` function.

    Per SYCL spec, group_barrier must perform both control barrier and memory
    fence operations. Hence, group_barrier requires two scopes and one memory
    consistency specification as its three arguments.

    mem_scope - scope of any memory consistency operations that are performed by
                the barrier. By default, mem_scope is set to `work_group`.
    exec_scope - scope that determines the set of work-items that synchronize at
                 barrier. Set to `work_group` for group_barrier always.
    spirv_memory_semantics_mask - Based on SYCL implementation. Always set to
                                  use sequential consistency memory order.
    """

    if not isinstance(group, GroupType):
        raise TypingError(
            "Expected a group should to be a GroupType value, but "
            f"encountered {type(group)}"
        )

    mem_scope = _get_memory_scope(fence_scope)
    # TODO: exec_scope needs to be determined based on
    # group argument. If group refers to a work_group then,
    # exec_scope is MemoryScope.WORK_GROUP.
    # If group is sub_group then, exec_scope needs to be
    # MemoryScope.SUB_GROUP
    exec_scope = get_scope(MemoryScope.WORK_GROUP.value)
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        MemoryOrder.SEQ_CST.value
    )

    def _ol_group_barrier_impl(
        group,
        fence_scope=MemoryScope.WORK_GROUP,
    ):  # pylint: disable=unused-argument
        # pylint: disable=no-value-for-parameter
        return _intrinsic_barrier(
            exec_scope, mem_scope, spirv_memory_semantics_mask
        )

    return _ol_group_barrier_impl
