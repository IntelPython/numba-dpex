# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as dpex
from numba_dpex.kernel_api import (
    AtomicRef,
    Item,
    MemoryOrder,
    MemoryScope,
    atomic_fence,
)


def test_atomic_fence():
    """A test for atomic_fence function."""

    @dpex.kernel
    def _kernel(item: Item, a, b):
        i = item.get_id(0)

        bref = AtomicRef(b, index=0)

        if i == 1:
            a[i] += 1
            atomic_fence(MemoryOrder.RELEASE, MemoryScope.DEVICE)
            bref.store(1)
        elif i == 0:
            while not bref.load():
                continue
            atomic_fence(MemoryOrder.ACQUIRE, MemoryScope.DEVICE)
            for idx in range(1, a.size):
                a[0] += a[idx]

    N = 2
    a = dpnp.ones(N, dtype=dpnp.int64)
    b = dpnp.zeros(1, dtype=dpnp.int64)

    dpex.call_kernel(_kernel, dpex.Range(N), a, b)

    assert a[0] == N + 1
