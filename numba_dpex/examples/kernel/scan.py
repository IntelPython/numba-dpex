# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""An implementation of the Hillis-Steele algorithm to compute prefix sums.

The algorithm is implemented to work with a single work group of N work items,
where N is the number of elements.
"""

import dpnp as np

import numba_dpex as ndpx
from numba_dpex import kernel_api as kapi

# 1D array size
N = 64


@ndpx.kernel
def kernel_hillis_steele_scan(nditem: kapi.NdItem, a, slm_b, slm_c):
    # Get local and global id and workgroup size
    gid = nditem.get_global_id(0)
    lid = nditem.get_local_id(0)
    ls = nditem.get_local_range(0)
    gr = nditem.get_group()

    # Initialize locals
    slm_c[lid] = slm_b[lid] = a[gid]

    kapi.group_barrier(gr)

    # Calculate prefix sum
    d = 1
    while d < ls:
        if lid > d:
            slm_c[lid] = slm_b[lid] + slm_b[lid - d]
        else:
            slm_c[lid] = slm_b[lid]

        kapi.group_barrier(gr)

        # Swap c and b
        e = slm_c[lid]
        slm_c[lid] = slm_b[lid]
        slm_b[lid] = e

        # Double the stride
        d *= 2

    kapi.group_barrier(gr, kapi.MemoryScope.DEVICE)

    a[gid] = slm_b[lid]


def main():
    arr = np.arange(N)
    print("Original array:", arr)

    print("Using device ...")
    print(arr.device)

    # Create temporals in local memory
    slm_b = kapi.LocalAccessor(N, dtype=arr.dtype)
    slm_c = kapi.LocalAccessor(N, dtype=arr.dtype)

    ndpx.call_kernel(
        kernel_hillis_steele_scan, ndpx.NdRange((N,), (N,)), arr, slm_b, slm_c
    )

    # the output should be [0, 1, 3, 6, ...]
    arr_np = np.asnumpy(arr)
    print(arr_np)

    print("Done...")


if __name__ == "__main__":
    main()
