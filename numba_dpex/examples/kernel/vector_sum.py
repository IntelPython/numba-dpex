# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The example demonstrates a 1D vector addition kernel.
"""

import dpnp
import numpy.testing as testing

import numba_dpex as ndpx


# Data parallel kernel implementing vector sum
@ndpx.kernel
def kernel_vector_sum(item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


# Utility function for printing and testing
def driver(a, b, c, global_size):
    ndpx.call_kernel(kernel_vector_sum, ndpx.Range(global_size), a, b, c)
    a_np = dpnp.asnumpy(a)  # Copy dpnp array a to NumPy array a_np
    b_np = dpnp.asnumpy(b)  # Copy dpnp array b to NumPy array b_np
    c_np = dpnp.asnumpy(c)  # Copy dpnp array c to NumPy array c_np
    testing.assert_equal(c_np, a_np + b_np)


# Main function
def main():
    N = 10
    global_size = N
    print("Vector size N", N)

    # Create random vectors on the default device
    a = dpnp.random.random(N)
    b = dpnp.random.random(N)
    c = dpnp.ones_like(a)

    print("Executing on device:")
    a.device.print_device_info()
    driver(a, b, c, global_size)
    print("Done...")


if __name__ == "__main__":
    main()
