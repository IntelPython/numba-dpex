# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The example demonstrates the use of :class:`numba_dpex.kernel_api.AtomicRef`.

The kernel shows the implementation of a reduction operation in numba-dpex
where every work-item is updating a global accumulator atomically.
"""
import dpnp

import numba_dpex as dpex
from numba_dpex import kernel_api as kapi


@dpex.kernel
def atomic_reduction(item: kapi.Item, a, res):
    """Array reduction using :func:`AtomicRef.fetch_add`.

    Args:
        item (kapi.Item): Index space id for each work item.
        a (dpnp.ndarray): An 1-d array to be reduced.
        res (dpnp.ndarray): A single element array into which the result is
            accumulated.
    """
    idx = item.get_id(0)
    acc = kapi.AtomicRef(res, 0)
    acc.fetch_add(a[idx])


def main():
    N = 1024

    a = dpnp.arange(0, N)
    res = dpnp.zeros(1, dtype=a.dtype)

    print("Executing on device:")
    a.device.print_device_info()

    dpex.call_kernel(atomic_reduction, dpex.Range(N), a, res)
    print(f"Summation of {N} integers = {res[0]}")

    assert res[0] == N * (N - 1) / 2


if __name__ == "__main__":
    main()
