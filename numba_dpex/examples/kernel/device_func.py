# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Demonstrates the usage of the :func:`numba_dpex.device_func` decorator.

Refer the API documentation and the Kenrel programming guide for further
details.
"""

import dpnp

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi


@dpex.device_func
def increment_by_1(a):
    """A device callable function that can be invoked from a kernel or
    another device function.
    """
    return a + 1


@dpex.device_func
def increment_and_sum_up(nd_item: kapi.NdItem, a):
    """Demonstrates the usage of group_barrier and NdItem usage in a
    device_func.
    """
    i = nd_item.get_global_id(0)

    a[i] += 1
    kapi.group_barrier(nd_item.get_group(), kapi.MemoryScope.DEVICE)

    if i == 0:
        for idx in range(1, a.size):
            a[0] += a[idx]


@dpex.kernel
def kernel1(item: kapi.Item, a, b):
    """Demonstrates calling a device function from a kernel."""
    i = item.get_id(0)
    b[i] = increment_by_1(a[i])


@dpex.kernel
def kernel2(nd_item: kapi.NdItem, a):
    """The kernel delegates everything to a device_func and calls it."""
    increment_and_sum_up(nd_item, a)


if __name__ == "__main__":
    # Array size
    N = 100
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    dpex.call_kernel(kernel1, dpex.Range(N), a, b)
    # b should be [2, 2, ...., 2]
    print(b)

    dpex.call_kernel(kernel2, dpex.NdRange((N,), (N,)), b)
    # b[0] should be 300
    print(b[0])
