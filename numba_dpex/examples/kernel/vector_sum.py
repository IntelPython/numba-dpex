# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import dpnp.random as dprandom
import numpy.testing as testing

import numba_dpex as ndpx

# Vector size
N = 10


# Data parallel kernel implementing vector sum
@ndpx.kernel
def kernel_vector_sum(a, b, c):
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


# Utility function for printing and testing
def driver(a, b, c, global_size):
    # Printing inputs
    print("A : ", a)
    print("B : ", b)

    # Invoking kernel
    kernel_vector_sum[global_size, ndpx.DEFAULT_LOCAL_SIZE](a, b, c)

    # Printing result
    print("A + B = ")
    print("C ", c)

    # Testing against NumPy
    a_np = a.asnumpy()  # Copy dpnp array a to NumPy array a_np
    b_np = b.asnumpy()  # Copy dpnp array a to NumPy array a_np
    testing.assert_equal(c, a_np + b_np)


# Main function
def main():
    global_size = N
    print("Vector size N", N)

    # Create random vectors on the default device
    a = dprandom.random(N)
    b = dprandom.random(N)
    c = dpnp.ones_like(a)

    print("Using device ...")
    print(a.device)
    driver(a, b, c, global_size)
    print("Done...")


if __name__ == "__main__":
    main()
