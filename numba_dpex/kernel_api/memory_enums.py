# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A collection of FlagEnum classes that syntactically represents the SYCL
memory enum classes.
"""

from numba_dpex.kernel_api.flag_enum import FlagEnum


class MemoryOrder(FlagEnum):
    """
    Analogue of :sycl_memory_order:`sycl::memory_order <>` enumeration.

    The integer values of the enums is kept consistent with the corresponding
    implementation in dpcpp.

    """

    RELAXED = 0
    ACQUIRE = 1
    CONSUME_UNSUPPORTED = 2
    RELEASE = 3
    ACQ_REL = 4
    SEQ_CST = 5


class MemoryScope(FlagEnum):
    """
    Analogue of :sycl_memory_scope:`sycl::memory_scope <>` enumeration.

    The integer values of the enums is kept consistent with the corresponding
    implementation in dpcpp.

    """

    WORK_ITEM = 0
    SUB_GROUP = 1
    WORK_GROUP = 2
    DEVICE = 3
    SYSTEM = 4


class AddressSpace(FlagEnum):
    """Analogue of :sycl_addr_space:`SYCL address space classes <>`.

    The integer values of the enums is kept consistent with the corresponding
    implementation in dpcpp.
    """

    PRIVATE = 0
    GLOBAL = 1
    CONSTANT = 2
    LOCAL = 3
    GENERIC = 4
