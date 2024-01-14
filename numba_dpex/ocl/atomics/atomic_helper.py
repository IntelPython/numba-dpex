# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto


class sycl_memory_order(Enum):
    """
    An enumeration of SYCL memory order. For more details please refer to
    SYCL 2020 specification, section 3.8.3.1
    (https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_memory_ordering).

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/sycl/ONEAPI/atomic_enums.hpp#L25

    =====================   ============
    Memory Order            Enum value
    =====================   ============
    relaxed                 0
    acquire                 1
    __consume_unsupported   2
    release                 3
    acq_rel                 4
    seq_cst                 5
    =====================   ============
    """

    relaxed = auto()
    acquire = auto()
    __consume_unsupported = auto()
    release = auto()
    acq_rel = auto()
    seq_cst = auto()


class sycl_memory_scope(Enum):
    """
    An enumeration of SYCL memory scope. For more details please refer to
    SYCL 2020 specification, section 3.8.3.2
    (https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_memory_scope).

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/sycl/ONEAPI/atomic_enums.hpp#L45

    ===============  ============
    Memory Scope     Enum value
    ===============  ============
    work_item        0
    sub_group        1
    work_group       2
    device           3
    system           4
    ===============  ============
    """

    work_item = auto()
    sub_group = auto()
    work_group = auto()
    device = auto()
    system = auto()


class _spv_scope(Enum):
    """
    An enumeration of SPIRV scope.

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/__spirv/spirv_types.hpp#L24
    """

    CrossDevice = 0
    Device = 1
    Workgroup = 2
    Subgroup = 3
    Invocation = 4


class _spv_memory_semantics_mask(Enum):
    """
    An enumeration of SPIRV memory semantics mask.

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/__spirv/spirv_types.hpp#L81
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
    This function translates SYCL memory order to SPIRV memory semantics mask.

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/sycl/detail/spirv.hpp#L220
    """

    spv_order = _spv_memory_semantics_mask.NONE.value
    if memory_order == sycl_memory_order.relaxed:
        spv_order = _spv_memory_semantics_mask.NONE.value
    elif memory_order == sycl_memory_order.__consume_unsupported:
        pass
    elif memory_order == sycl_memory_order.acquire:
        spv_order = _spv_memory_semantics_mask.Acquire.value
    elif memory_order == sycl_memory_order.release:
        spv_order = _spv_memory_semantics_mask.Release.value
    elif memory_order == sycl_memory_order.acq_rel:
        spv_order = _spv_memory_semantics_mask.AcquireRelease.value
    elif memory_order == sycl_memory_order.seq_cst:
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
    This function translates SYCL memory scope to SPIRV scope.

    For DPCPP implementation please refer:
    https://github.com/intel/llvm/blob/sycl-nightly/20210507/sycl/include/CL/sycl/detail/spirv.hpp#L247
    """

    if memory_scope == sycl_memory_scope.work_item:
        return _spv_scope.Invocation.value
    elif memory_scope == sycl_memory_scope.sub_group:
        return _spv_scope.Subgroup.value
    elif memory_scope == sycl_memory_scope.work_group:
        return _spv_scope.Workgroup.value
    elif memory_scope == sycl_memory_scope.device:
        return _spv_scope.Device.value
    elif memory_scope == sycl_memory_scope.system:
        return _spv_scope.CrossDevice.value
    else:
        raise ValueError("Invalid memory scope provided")
