# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Helper module to generate LLVM IR SPIR-V intrinsic calls.
"""
from enum import IntEnum

from numba.core import types

from numba_dpex.kernel_api import MemoryOrder, MemoryScope


class _SpvScope(IntEnum):
    """
    An enumeration to store the dpcpp values for SPIR-V memory scope.
    """

    CROSS_DEVICE = 0
    DEVICE = 1
    WORKGROUP = 2
    SUBGROUP = 3
    INVOCATION = 4


class _SpvMemorySemanticsMask(IntEnum):
    """
    An enumeration to store the dpcpp values for SPIR-V mask.

    """

    NONE = 0x0
    ACQUIRE = 0x2
    RELEASE = 0x4
    ACQUIRE_RELEASE = 0x8
    SEQUENTIALLY_CONSISTENT = 0x10
    UNIFORM_MEMORY = 0x40
    SUBGROUP_MEMORY = 0x80
    WORKGROUP_MEMORY = 0x100
    CROSS_WORKGROUP_MEMORY = 0x200
    ATOMIC_COUNTER_MEMORY = 0x400
    IMAGE_MEMORY = 0x800


_spv_atomic_instructions_map = {
    "fetch_add": {
        types.int32: "__spirv_AtomicIAdd",
        types.int64: "__spirv_AtomicIAdd",
        types.float32: "__spirv_AtomicFAddEXT",
        types.float64: "__spirv_AtomicFAddEXT",
    },
    "fetch_sub": {
        types.int32: "__spirv_AtomicISub",
        types.int64: "__spirv_AtomicISub",
        types.float32: "__spirv_AtomicFSubEXT",
        types.float64: "__spirv_AtomicFSubEXT",
    },
    "fetch_min": {
        types.int32: "__spirv_AtomicSMin",
        types.int64: "__spirv_AtomicSMin",
        types.float32: "__spirv_AtomicFMinEXT",
        types.float64: "__spirv_AtomicFMinEXT",
    },
    "fetch_max": {
        types.int32: "__spirv_AtomicSMax",
        types.int64: "__spirv_AtomicSMax",
        types.float32: "__spirv_AtomicFMaxEXT",
        types.float64: "__spirv_AtomicFMaxEXT",
    },
    "fetch_and": {
        types.int32: "__spirv_AtomicAnd",
        types.int64: "__spirv_AtomicAnd",
    },
    "fetch_or": {
        types.int32: "__spirv_AtomicOr",
        types.int64: "__spirv_AtomicOr",
    },
    "fetch_xor": {
        types.int32: "__spirv_AtomicXor",
        types.int64: "__spirv_AtomicXor",
    },
}


def get_atomic_inst_name(atomic_inst: str, atomic_ref_dtype: types.Type) -> str:
    """Returns a string corresponding to the LLVM IR intrinsic function
    generated for a particular atomic operation for a specific data type.

    Args:
        atomic_inst (str): The string name of an atomic operation to look up.
        atomic_ref_dtype (numba.core.types.Type): The Numba type object
        corresponding to the dtype of the object stored in an AtomicRef.

    Raises:
        ValueError: If the atomic operation is not found in the
        ``_spv_atomic_instructions_map`` dictionary.
        ValueError: If no instruction is found for a combination of the
        atomic operation and the dtype in the ``_spv_atomic_instructions_map``
        dictionary.

    Returns:
        str: A string corresponding to the LLVM IR intrinsic that should be
        generated for the atomic operation for the specified dtype.
    """
    inst_ref_types_map = _spv_atomic_instructions_map.get(atomic_inst)
    if inst_ref_types_map is None:
        raise ValueError("Unsupported atomic instruction")

    inst_name = inst_ref_types_map.get(atomic_ref_dtype)

    if inst_name is None:
        raise ValueError(
            "Unsupported atomic reference type for instruction " + atomic_inst
        )
    return inst_name


def get_memory_semantics_mask(memory_order):
    """
    Equivalent of the getMemorySemanticsMask function in dpcpp's
    sycl/include/sycl/detail/spirv.hpp.  Translates SYCL memory order to a
    SPIR-V  memory semantics mask.

    """

    spv_order = _SpvMemorySemanticsMask.NONE.value

    if memory_order == MemoryOrder.RELAXED.value:
        spv_order = _SpvMemorySemanticsMask.NONE.value
    elif memory_order == MemoryOrder.CONSUME_UNSUPPORTED.value:
        pass
    elif memory_order == MemoryOrder.ACQUIRE.value:
        spv_order = _SpvMemorySemanticsMask.ACQUIRE.value
    elif memory_order == MemoryOrder.RELEASE.value:
        spv_order = _SpvMemorySemanticsMask.RELEASE.value
    elif memory_order == MemoryOrder.ACQ_REL.value:
        spv_order = _SpvMemorySemanticsMask.ACQUIRE_RELEASE.value
    elif memory_order == MemoryOrder.SEQ_CST.value:
        spv_order = _SpvMemorySemanticsMask.SEQUENTIALLY_CONSISTENT.value
    else:
        raise ValueError("Invalid memory order provided")

    return (
        spv_order
        | _SpvMemorySemanticsMask.SUBGROUP_MEMORY.value
        | _SpvMemorySemanticsMask.WORKGROUP_MEMORY.value
        | _SpvMemorySemanticsMask.CROSS_WORKGROUP_MEMORY.value
    )


def get_scope(memory_scope):
    """
    Translates a memory scope enum value to an equivalent SPIR-V memory scope
    value.
    """
    retval = None
    if memory_scope == MemoryScope.WORK_ITEM.value:
        retval = _SpvScope.INVOCATION.value
    elif memory_scope == MemoryScope.SUB_GROUP.value:
        retval = _SpvScope.SUBGROUP.value
    elif memory_scope == MemoryScope.WORK_GROUP.value:
        retval = _SpvScope.WORKGROUP.value
    elif memory_scope == MemoryScope.DEVICE.value:
        retval = _SpvScope.DEVICE.value
    elif memory_scope == MemoryScope.SYSTEM.value:
        retval = _SpvScope.CROSS_DEVICE.value
    else:
        raise ValueError("Invalid memory scope provided")

    return retval
