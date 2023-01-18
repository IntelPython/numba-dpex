# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpex

# Array size
N = 10


# A device callable function that can be invoked from ``kernel`` and other
# device functions
@ndpex.func
def a_device_function(a):
    return a + 1


# A device callable function can call another device function
@ndpex.func
def another_device_function(a):
    return a_device_function(a * 2)


# A kernel function that calls the device function
@ndpex.kernel
def a_kernel_function(a, b):
    i = ndpex.get_global_id(0)
    b[i] = another_device_function(a[i])


# Utility function for printing
def driver(a, b, N):
    print("A=", a)
    a_kernel_function[N](a, b)
    print("B=", b)


# Main function
def main():
    a = np.ones(N)
    b = np.ones(N)

    print("Using device ...")
    print(a.device)
    driver(a, b, N)

    print("Done...")


if __name__ == "__main__":
    main()
