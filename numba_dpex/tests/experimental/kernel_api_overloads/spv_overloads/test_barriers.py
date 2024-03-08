# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.kernel_api import MemoryScope, NdItem, group_barrier


def test_group_barrier():
    """A test for group_barrier function."""

    @dpex_exp.kernel
    def _kernel(nd_item: NdItem, a):
        i = nd_item.get_global_id(0)

        a[i] += 1
        group_barrier(nd_item.get_group(), MemoryScope.DEVICE)

        if i == 0:
            for idx in range(1, a.size):
                a[0] += a[idx]

    N = 16
    a = dpnp.ones(N, dtype=dpnp.int32)

    dpex_exp.call_kernel(_kernel, dpex.NdRange((N,), (N,)), a)

    assert a[0] == N * 2
