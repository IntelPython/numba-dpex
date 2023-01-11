# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpex
from numba_dpex import float32, int32, int64

# Array size
N = 10


# A device callable function that can be invoked from ``kernel`` and other device functions
@ndpex.func  # ([int32(int32), int64(int64)])
def a_device_function(a):
    return a + 1


# A device callable function can call another device function
# @ndpex.func
# def another_device_function(a):
#     return a_device_function(a * 2)


# A kernel function that calls the device function
@ndpex.kernel(enable_cache=True)
def a_kernel_function(a, b):
    i = ndpex.get_global_id(0)
    b[i] = a_device_function(a[i])


# @ndpex.kernel(enable_cache=True)
# def another_kernel_function(a, b):
#     i = ndpex.get_global_id(0)
#     b[i] = a_device_function(a[i])


# Utility function for printing
# def driver(a, b, N):
#     print("A=", a)
#     a_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)
#     print("B=", b)


# Main function
# def main():

a = np.ones(N, dtype=np.int32)
b = np.ones(N, dtype=np.int32)
a_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)
print("int32")
print("a =", a)
print("b =", b)


# a = np.ones(N, dtype=np.int64)
# b = np.ones(N, dtype=np.int64)
# a_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)
# print("int64")
# print("a =", a)
# print("b =", b)


# a = np.ones(N, dtype=np.float32)
# b = np.ones(N, dtype=np.float32)
# a_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)
# print("float32")
# print("a =", a)
# print("b =", b)


# another_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)
# a_kernel_function[N, ndpex.DEFAULT_LOCAL_SIZE](a, b)

# print("Using device ...")
# print(a.device)
# driver(a, b, N)

# print("Done...")


# main()
# if __name__ == "__main__":
#     main()
