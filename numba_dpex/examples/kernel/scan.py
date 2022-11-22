# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

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
    ndpx.barrier(NDPXK_LOCAL_MEM_FENCE)

    # Calculate prefix sum
    d = 1
    while d < ls:
        if lid > d:
            c[lid] = b[lid] + b[lid - d]
        else:
            c[lid] = b[lid]

        ndpx.barrier(NDPXK_LOCAL_MEM_FENCE)

        # Swap c and b
        e = c[lid]
        c[lid] = b[lid]
        b[lid] = e

        # Double the stride
        d *= 2

    ndpx.barrier()  # NDPXK_GLOBAL_MEM_FENCE
    a[gid] = b[lid]

def main():
    arr = np.arange(N)
    res = np.empty(N)
    print("Original array:", arr)

    print("Using device ...")
    arr.device.print_device_info()
    kernel_hillis_steele_scan[N, ndpx.DEFAULT_LOCAL_SIZE](arr)

    # the output should be [0, 1, 3, 6, ...]
    print(arr)

    print("Done...")


if __name__ == "__main__":
    main()
