# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex


@dpex.func
def a_device_function(a):
    """
    A ``func`` is a device callable function that can be invoked from
    ``kernel`` and other ``func`` functions.
    """
    return a + 1


@dpex.func
def another_device_function(a):
    return a_device_function(a)


@dpex.kernel
def a_kernel_function(a, b):
    i = dpex.get_global_id(0)
    b[i] = another_device_function(a[i])


def driver(a, b, N):
    print(b)
    print("--------")
    a_kernel_function[N, dpex.DEFAULT_LOCAL_SIZE](a, b)
    print(b)


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        driver(a, b, N)

    print("Done...")


if __name__ == "__main__":
    main()
