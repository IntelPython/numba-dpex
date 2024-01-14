# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# scan.py is not working due to issue: https://github.com/IntelPython/numba-dpex/issues/829

import dpnp as np

import numba_dpex as ndpx

# 1D array size
N = 64


# Implements Hillis-Steele prefix sum algorithm
@ndpx.kernel
def kernel_hillis_steele_scan(a):
    # Get local and global id and workgroup size
    gid = ndpx.get_global_id(0)
    lid = ndpx.get_local_id(0)
    ls = ndpx.get_local_size(0)

    # Create temporals in local memory
    b = ndpx.local.array(ls, dtype=a.dtype)
    c = ndpx.local.array(ls, dtype=a.dtype)

    # Initialize locals
    c[lid] = b[lid] = a[gid]
    ndpx.barrier(ndpx.LOCAL_MEM_FENCE)

    # Calculate prefix sum
    d = 1
    while d < ls:
        if lid > d:
            c[lid] = b[lid] + b[lid - d]
        else:
            c[lid] = b[lid]

        ndpx.barrier(ndpx.LOCAL_MEM_FENCE)

        # Swap c and b
        e = c[lid]
        c[lid] = b[lid]
        b[lid] = e

        # Double the stride
        d *= 2

    ndpx.barrier()  # The same as ndpx.barrier(ndpx.GLOBAL_MEM_FENCE)
    a[gid] = b[lid]


def main():
    arr = np.arange(N)
    print("Original array:", arr)

    print("Using device ...")
    print(arr.device)
    kernel_hillis_steele_scan[ndpx.Range(N)](arr)

    # the output should be [0, 1, 3, 6, ...]
    arr_np = np.asnumpy(arr)
    print(arr_np)

    print("Done...")


if __name__ == "__main__":
    main()
