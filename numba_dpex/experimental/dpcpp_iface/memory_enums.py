# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba_dpex.experimental.flag_enum import FlagEnum


class MemoryOrder(FlagEnum):
    """
    An enumeration of the supported ``sycl::memory_order`` values in dpcpp. The
    integer values of the enums is kept consistent with the corresponding
    implementation in dpcpp.


    =====================   ============
    Order                   Enum value
    =====================   ============
    relaxed                 0
    acquire                 1
    consume_unsupported     2
    release                 3
    acq_rel                 4
    seq_cst                 5
    =====================   ============
    """

    relaxed = 0
    acquire = 1
    consume_unsupported = 2
    release = 3
    acq_rel = 4
    seq_cst = 5


class MemoryScope(FlagEnum):
    """
    An enumeration of SYCL memory scope. For more details please refer to
    SYCL 2020 specification, section 3.8.3.2

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

    work_item = 0
    sub_group = 1
    work_group = 2
    device = 3
    system = 4


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
