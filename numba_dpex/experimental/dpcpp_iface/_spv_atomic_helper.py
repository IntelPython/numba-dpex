# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from numba.core import types

from .memory_enums import MemoryOrder, MemoryScope


class _spv_scope(IntEnum):
    """
    An enumeration to store the dpcpp values for spirv scope.
    """

    CrossDevice = 0
    Device = 1
    Workgroup = 2
    Subgroup = 3
    Invocation = 4


class _spv_memory_semantics_mask(IntEnum):
    """
    An enumeration to store the dpcpp values for spirv mask.

    """

    NONE = 0x0
    Acquire = 0x2
    Release = 0x4
    AcquireRelease = 0x8
    SequentiallyConsistent = 0x10
    UniformMemory = 0x40
    SubgroupMemory = 0x80
    WorkgroupMemory = 0x100
    CrossWorkgroupMemory = 0x200
    AtomicCounterMemory = 0x400
    ImageMemory = 0x800


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


def get_atomic_inst_name(atomic_inst, atomic_ref_dtype):
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
    Translates SYCL memory order to SPIRV memory semantics mask, based on the
    getMemorySemanticsMask function in dpcpp's
    sycl/include/sycl/detail/spirv.hpp.


    """

    spv_order = _spv_memory_semantics_mask.NONE.value

    if memory_order == MemoryOrder.relaxed.value:
        spv_order = _spv_memory_semantics_mask.NONE.value
    elif memory_order == MemoryOrder.consume_unsupported.value:
        pass
    elif memory_order == MemoryOrder.acquire.value:
        spv_order = _spv_memory_semantics_mask.Acquire.value
    elif memory_order == MemoryOrder.release.value:
        spv_order = _spv_memory_semantics_mask.Release.value
    elif memory_order == MemoryOrder.acq_rel.value:
        spv_order = _spv_memory_semantics_mask.AcquireRelease.value
    elif memory_order == MemoryOrder.seq_cst.value:
        spv_order = _spv_memory_semantics_mask.SequentiallyConsistent.value
    else:
        raise ValueError("Invalid memory order provided")

    return (
        spv_order
        | _spv_memory_semantics_mask.SubgroupMemory.value
        | _spv_memory_semantics_mask.WorkgroupMemory.value
        | _spv_memory_semantics_mask.CrossWorkgroupMemory.value
    )


def get_scope(memory_scope):
    """
    Translates SYCL memory scope to SPIRV scope.
    """

    if memory_scope == MemoryScope.work_item.value:
        return _spv_scope.Invocation.value
    elif memory_scope == MemoryScope.sub_group.value:
        return _spv_scope.Subgroup.value
    elif memory_scope == MemoryScope.work_group.value:
        return _spv_scope.Workgroup.value
    elif memory_scope == MemoryScope.device.value:
        return _spv_scope.Device.value
    elif memory_scope == MemoryScope.system.value:
        return _spv_scope.CrossDevice.value
    else:
        raise ValueError("Invalid memory scope provided")
