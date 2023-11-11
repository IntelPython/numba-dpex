# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

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
