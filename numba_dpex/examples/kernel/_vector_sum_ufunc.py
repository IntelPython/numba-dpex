# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# This test is disabled in CI since dpnp array a is not supporting a.asnumpy().
###########################################################################

import dpnp.random as dprandom
import numpy.testing as testing

import numba_dpex as ndpx

# Vector size
N = 10


# Data parallel kernel implementing vector sum as ufunc
@ndpx.vectorize
def ufunc_sum(a, b):
    return a + b


# Main function
def main():
    print("Vector size N", N)
    # Create random vectors on the default device
    a = dprandom.random(N)
    b = dprandom.random(N)

    # Printing inputs
    print("A : ", a)
    print("B : ", b)
    print("Using device ...")
    print(a.device)

    # Invoking kernel
    c = ufunc_sum(a, b)

    # Printing result
    print("A + B = ")
    print("C ", c)

    # Testing against NumPy
    a_np = a.asnumpy()  # Copy dpnp array a to NumPy array a_np
    b_np = b.asnumpy()  # Copy dpnp array a to NumPy array a_np
    testing.assert_equal(c, a_np + b_np)

    print("Done...")


if __name__ == "__main__":
    main()
