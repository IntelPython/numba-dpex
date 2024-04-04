# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides overloads for functions included in kernel_api.atomic_fence
that generate dpcpp SPIR-V LLVM IR intrinsic function calls.
"""
from llvmlite import ir as llvmir
from numba.core import types
from numba.extending import intrinsic, overload

from numba_dpex.kernel_api import atomic_fence

from ..target import SPIRV_TARGET_NAME
from ._spv_atomic_inst_helper import get_memory_semantics_mask, get_scope
from .spv_fn_declarations import (
    _SUPPORT_CONVERGENT,
    get_or_insert_spv_atomic_fence_fn,
)


@intrinsic(target=SPIRV_TARGET_NAME)
def _intrinsic_atomic_fence(
    ty_context, ty_spirv_mem_sem_mask, ty_spirv_scope
):  # pylint: disable=unused-argument

    # Signature of `__spirv_MemoryBarrier` call that is
    # generated for atomic_fence. It takes two arguments -
    # scope and memory_semantics_mask.
    # All arguments have to be of type unsigned int32.
    sig = types.void(types.uint32, types.uint32)

    def _intrinsic_atomic_fence_gen(
        context, builder, sig, args
    ):  # pylint: disable=unused-argument
        callinst = builder.call(
            get_or_insert_spv_atomic_fence_fn(builder.module),
            [
                builder.trunc(args[1], llvmir.IntType(32)),  # scope
                builder.trunc(args[0], llvmir.IntType(32)),  # semantics mask
            ],
        )

        if _SUPPORT_CONVERGENT:  # pylint: disable=duplicate-code
            callinst.attributes.add("convergent")
        callinst.attributes.add("nounwind")

    return (
        sig,
        _intrinsic_atomic_fence_gen,
    )


@overload(
    atomic_fence,
    prefer_literal=True,
    target=SPIRV_TARGET_NAME,
)
def ol_atomic_fence(memory_order, memory_scope):
    """SPIR-V overload for
    :meth:`numba_dpex.kernel_api.atomic_fence`.

    Generates the same LLVM IR instruction as DPC++ for the SYCL
    `atomic_fence` function.
    """
    spirv_memory_semantics_mask = get_memory_semantics_mask(
        memory_order.literal_value
    )
    spirv_scope = get_scope(memory_scope.literal_value)

    def ol_atomic_fence_impl(
        memory_order, memory_scope
    ):  # pylint: disable=unused-argument
        # pylint: disable=no-value-for-parameter
        return _intrinsic_atomic_fence(spirv_memory_semantics_mask, spirv_scope)

    return ol_atomic_fence_impl
