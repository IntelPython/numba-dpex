# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
numba_dpex can run several of the numpy.random RNG functions called inside
a JIT function on a SYCL device using dpnp
(https://github.com/IntelPython/dpnp). As with the rest of numba_dpex examples,
this feature is also available by simply invoking a ``numba.jit`` function with
the numpy.random calls from within a ``dpctl.device_context``scope.
"""

import dpctl
import numba
import numpy as np


@numba.njit
def rand():
    return np.random.rand(3, 2)


@numba.njit
def random_sample(size):
    return np.random.random_sample(size)


@numba.njit
def random_exponential(scale, size):
    return np.random.exponential(scale, size)


@numba.njit
def random_normal(loc, scale, size):
    return np.random.normal(loc, scale, size)


def main():
    size = 9
    scale = 3.0

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        result = rand()
        # Random values in a given shape (3, 2)
        print(result)

        result = random_sample(size)
        # Array of shape (9,) with random floats in the
        # half-open interval [0.0, 1.0)
        print(result)

        result = random_exponential(scale, size)
        # Array of shape (9,) with samples from an exponential distribution
        print(result)

        result = random_normal(0.0, 0.1, size)
        # Array of shape (9,) with samples from a normal distribution
        print(result)

    print("Done...")


if __name__ == "__main__":
    main()
