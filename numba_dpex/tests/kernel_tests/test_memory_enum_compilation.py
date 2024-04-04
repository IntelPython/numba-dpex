# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as dpex
from numba_dpex import Range
from numba_dpex.kernel_api import AddressSpace, MemoryOrder, MemoryScope


def test_compilation_of_memory_order():
    """Tests if a MemoryOrder flags can be used inside a kernel function."""

    @dpex.kernel
    def store_memory_order_flag(a):
        a[0] = MemoryOrder.RELAXED
        a[1] = MemoryOrder.CONSUME_UNSUPPORTED
        a[2] = MemoryOrder.ACQ_REL
        a[3] = MemoryOrder.ACQUIRE
        a[4] = MemoryOrder.RELEASE
        a[5] = MemoryOrder.SEQ_CST

    a = dpnp.ones(10, dtype=dpnp.int64)
    dpex.call_kernel(store_memory_order_flag, Range(10), a)

    assert a[0] == MemoryOrder.RELAXED
    assert a[1] == MemoryOrder.CONSUME_UNSUPPORTED
    assert a[2] == MemoryOrder.ACQ_REL
    assert a[3] == MemoryOrder.ACQUIRE
    assert a[4] == MemoryOrder.RELEASE
    assert a[5] == MemoryOrder.SEQ_CST


def test_compilation_of_memory_scope():
    """Tests if a MemoryScope flags can be used inside a kernel function."""

    @dpex.kernel
    def store_memory_scope_flag(a):
        a[0] = MemoryScope.DEVICE
        a[1] = MemoryScope.SUB_GROUP
        a[2] = MemoryScope.WORK_GROUP
        a[3] = MemoryScope.SYSTEM
        a[4] = MemoryScope.WORK_ITEM

    a = dpnp.ones(10, dtype=dpnp.int64)
    dpex.call_kernel(store_memory_scope_flag, Range(10), a)

    assert a[0] == MemoryScope.DEVICE
    assert a[1] == MemoryScope.SUB_GROUP
    assert a[2] == MemoryScope.WORK_GROUP
    assert a[3] == MemoryScope.SYSTEM
    assert a[4] == MemoryScope.WORK_ITEM


def test_compilation_of_address_space():
    """Tests if a AddressSpace flags can be used inside a kernel function."""

    @dpex.kernel
    def store_address_space_flag(a):
        a[0] = AddressSpace.CONSTANT
        a[1] = AddressSpace.GENERIC
        a[2] = AddressSpace.GLOBAL
        a[3] = AddressSpace.LOCAL
        a[4] = AddressSpace.PRIVATE

    a = dpnp.ones(10, dtype=dpnp.int64)
    dpex.call_kernel(store_address_space_flag, Range(10), a)

    assert a[0] == AddressSpace.CONSTANT
    assert a[1] == AddressSpace.GENERIC
    assert a[2] == AddressSpace.GLOBAL
    assert a[3] == AddressSpace.LOCAL
    assert a[4] == AddressSpace.PRIVATE
