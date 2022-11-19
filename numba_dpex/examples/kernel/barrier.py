# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp as np
from numba import float32

import numba_dpex as dpex


# This example demonstrates the usage of numba_dpex's ``barrier``
# intrinsic function. The ``barrier`` function is usable only inside
# a ``kernel`` and is equivalent to OpenCL's ``barrier`` function.
def no_arg_barrier_support():
    N = 10  # 1D array size

    @dpex.kernel
    def kernel_twice(a):
        i = dpex.get_global_id(0)
        d = a[i]

        dpex.barrier()  # No argument defaults to global mem fence

        a[i] = d * 2

    arr = np.arange(N, dtype=np.float32)
    print("Original array:", arr)

    print("Using device ...")
    arr.device.print_device_info()

    kernel_twice[N, dpex.DEFAULT_LOCAL_SIZE](arr)

    # the output should be [0, 2, 4, 6, ...]
    print(arr)


# This example demonstrates the usage of numba-dpex's ``local.array``
# intrinsic function. The function is used to create a static array
# allocated on the devices local address space.
def local_memory():
    N = 10  # 1D array size

    @dpex.kernel
    def reverse_array(a):
        lm = dpex.local.array(shape=N, dtype=float32)
        i = dpex.get_global_id(0)

        # preload
        lm[i] = a[i]

        # barrier local or global will both work as we only have one work group
        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)  # local mem fence

        # write
        a[i] += lm[N - 1 - i]

    arr = np.arange(N, dtype=np.float32)
    print("Original array:", arr)

    print("Using device ...")
    arr.device.print_device_info()

    reverse_array[N, dpex.DEFAULT_LOCAL_SIZE](arr)

    # the output should be [9, 8, 7, ...]
    print(arr)


def main():
    no_arg_barrier_support()
    local_memory()

    print("Done...")


if __name__ == "__main__":
    main()
