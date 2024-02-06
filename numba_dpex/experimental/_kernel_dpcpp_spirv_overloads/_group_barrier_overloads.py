# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides overloads for functions included in kernel_iface.barrier that
generate dpcpp SPIR-V LLVM IR intrinsic function calls.
"""
import warnings

from llvmlite import ir as llvmir
from numba.core import cgutils, types
from numba.core.errors import TypingError
from numba.extending import intrinsic, overload

from numba_dpex.core import itanium_mangler as ext_itanium_mangler
from numba_dpex.experimental.core.types.kernel_api.items import GroupType
from numba_dpex.experimental.target import DPEX_KERNEL_EXP_TARGET_NAME
from numba_dpex.kernel_api import group_barrier
from numba_dpex.kernel_api.memory_enums import MemoryOrder, MemoryScope

from ._spv_atomic_inst_helper import get_memory_semantics_mask, get_scope

_SUPPORT_CONVERGENT = True

try:
    llvmir.FunctionAttributes("convergent")
except ValueError:
    warnings.warn(
        "convergent attribute is supported only starting llvmlite "
        + "0.42. Not setting this attribute may result in unexpected behavior"
        + "when using group_barrier"
    )
    _SUPPORT_CONVERGENT = False


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

        if _SUPPORT_CONVERGENT:
            fn.attributes.add("convergent")
        fn.attributes.add("nounwind")
        fn.calling_convention = "spir_func"

        callinst = builder.call(fn, fn_args)

        if _SUPPORT_CONVERGENT:
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
