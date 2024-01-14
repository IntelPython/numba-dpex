# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A collection of FlagEnum classes that syntactically represents the SYCL
memory enum classes.
"""

from numba_dpex.experimental.flag_enum import FlagEnum


class MemoryOrder(FlagEnum):
    """
    An enumeration of the supported ``sycl::memory_order`` values in dpcpp. The
    integer values of the enums is kept consistent with the corresponding
    implementation in dpcpp.

    =====================   ============
    Order                   Enum value
    =====================   ============
    RELAXED                 0
    ACQUIRE                 1
    CONSUME_UNSUPPORTED     2
    RELEASE                 3
    ACQ_REL                 4
    SEQ_CST                 5
    =====================   ============
    """

    RELAXED = 0
    ACQUIRE = 1
    CONSUME_UNSUPPORTED = 2
    RELEASE = 3
    ACQ_REL = 4
    SEQ_CST = 5


class MemoryScope(FlagEnum):
    """
    An enumeration of SYCL memory scope. For more details please refer to
    SYCL 2020 specification, section 3.8.3.2

    ===============  ============
    Memory Scope     Enum value
    ===============  ============
    WORK_ITEM        0
    SUB_GROUP        1
    WORK_GROUP       2
    DEVICE           3
    SYSTEM           4
    ===============  ============
    """

    WORK_ITEM = 0
    SUB_GROUP = 1
    WORK_GROUP = 2
    DEVICE = 3
    SYSTEM = 4


class AddressSpace(FlagEnum):
    """The address space values supported by numba_dpex.

    ==================   ============
    Address space        Value
    ==================   ============
    PRIVATE              0
    GLOBAL               1
    CONSTANT             2
    LOCAL                3
    GENERIC              4
    ==================   ============
    """

    PRIVATE = 0
    GLOBAL = 1
    CONSTANT = 2
    LOCAL = 3
    GENERIC = 4
