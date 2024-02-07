# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
import numpy.testing as testing

import numba_dpex as ndpx
import numba_dpex.experimental as ndpx_ex


def vector_sum(a, b, c):
    i = ndpx.get_global_id(0)
    c[i] = a[i] + b[i]


# Utility function for printing and testing
def driver(a, b, c, global_size):
    # Sim result
    c_sim = dpnp.zeros_like(c)

    # Call sim kernel
    ndpx_ex.call_kernel(vector_sum, ndpx.Range(global_size), a, b, c_sim)

    # Call dpex kernel
    ndpx_ex.call_kernel(
        ndpx_ex.kernel(vector_sum), ndpx.Range(global_size), a, b, c
    )

    # Compare kernel result with simulator
    testing.assert_equal(c.asnumpy(), c_sim.asnumpy())


# Main function
def main():
    N = 10
    global_size = N
    print("Vector size N", N)

    # Create random vectors on the default device
    a = dpnp.random.random(N)
    b = dpnp.random.random(N)
    c = dpnp.ones_like(a)

    print("Using device ...")
    print(a.device)
    driver(a, b, c, global_size)
    print("Done...")


if __name__ == "__main__":
    main()
