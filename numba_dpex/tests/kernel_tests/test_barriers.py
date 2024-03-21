# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as dpex
from numba_dpex.kernel_api import MemoryScope, NdItem, group_barrier


def test_group_barrier():
    """A test for group_barrier function."""

    @dpex.kernel
    def _kernel(nd_item: NdItem, a):
        i = nd_item.get_global_id(0)

        a[i] += 1
        group_barrier(nd_item.get_group(), MemoryScope.DEVICE)

        if i == 0:
            for idx in range(1, a.size):
                a[0] += a[idx]

    N = 16
    a = dpnp.ones(N, dtype=dpnp.int32)

    dpex.call_kernel(_kernel, dpex.NdRange((N,), (N,)), a)

    assert a[0] == N * 2


def test_group_barrier_device_func():
    """A test for group_barrier function."""

    @dpex.device_func
    def _increment_value(nd_item: NdItem, a):
        i = nd_item.get_global_id(0)

        a[i] += 1
        group_barrier(nd_item.get_group(), MemoryScope.DEVICE)

        if i == 0:
            for idx in range(1, a.size):
                a[0] += a[idx]

    @dpex.kernel
    def _kernel(nd_item: NdItem, a):
        _increment_value(nd_item, a)

    N = 16
    a = dpnp.ones(N, dtype=dpnp.int32)

    dpex.call_kernel(_kernel, dpex.NdRange((N,), (N,)), a)

    assert a[0] == N * 2
