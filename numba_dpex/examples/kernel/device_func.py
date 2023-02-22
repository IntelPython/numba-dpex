# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

import numba_dpex as ndpx
from numba_dpex import float32, int32, int64

# Array size
N = 10


# A device callable function that can be invoked from
# ``kernel`` and other device functions
@ndpx.func
def a_device_function(a):
    return a + 1


# A device callable function with signature that can be invoked
# from ``kernel`` and other device functions
@ndpx.func(int32(int32))
def a_device_function_int32(a):
    return a + 1


# A device callable function with list signature that can be invoked
# from ``kernel`` and other device functions
@ndpx.func([int32(int32), float32(float32)])
def a_device_function_int32_float32(a):
    return a + 1


# A device callable function can call another device function
@ndpx.func
def another_device_function(a):
    return a_device_function(a * 2)


# A kernel function that calls the device function
@ndpx.kernel
def a_kernel_function(a, b):
    i = ndpx.get_global_id(0)
    b[i] = another_device_function(a[i])


# A kernel function that calls the device function
@ndpx.kernel
def a_kernel_function_int32(a, b):
    i = ndpx.get_global_id(0)
    b[i] = a_device_function_int32(a[i])


# A kernel function that calls the device function
@ndpx.kernel
def a_kernel_function_int32_float32(a, b):
    i = ndpx.get_global_id(0)
    b[i] = a_device_function_int32_float32(a[i])


# test function 1: tests basic
def test1():
    a = np.ones(N)
    b = np.ones(N)

    print("Using device ...")
    print(a.device)

    print("A=", a)
    try:
        a_kernel_function[ndpx.Range(N)](a, b)
    except Exception as err:
        print(err)
    print("B=", b)

    print("Done...")


# test function 2: test device func with signature
def test2():
    a = np.ones(N, dtype=np.int32)
    b = np.ones(N, dtype=np.int32)

    print("Using device ...")
    print(a.device)

    print("A=", a)
    try:
        a_kernel_function_int32[ndpx.Range(N)](a, b)
    except Exception as err:
        print(err)
    print("B=", b)

    print("Done...")


# test function 3: test device function with list signature
def test3():
    a = np.ones(N, dtype=np.int32)
    b = np.ones(N, dtype=np.int32)

    print("Using device ...")
    print(a.device)

    print("A=", a)
    try:
        a_kernel_function_int32_float32[ndpx.Range(N)](a, b)
    except Exception as err:
        print(err)
    print("B=", b)

    # with a different dtype
    a = np.ones(N, dtype=np.float32)
    b = np.ones(N, dtype=np.float32)

    print("Using device ...")
    print(a.device)

    print("A=", a)
    try:
        a_kernel_function_int32_float32[ndpx.Range(N)](a, b)
    except Exception as err:
        print(err)
    print("B=", b)

    # this will fail, since int64 is not in
    # the signature list: [int32(int32), float32(float32)]
    a = np.ones(N, dtype=np.int64)
    b = np.ones(N, dtype=np.int64)

    print("Using device ...")
    print(a.device)

    print("A=", a)
    try:
        a_kernel_function_int32_float32[ndpx.Range(N)](a, b)
    except Exception as err:
        print(err)
    print("B=", b)

    print("Done...")


# main function
if __name__ == "__main__":
    test1()
    test2()
    test3()
