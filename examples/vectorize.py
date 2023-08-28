# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
from numba import float64, vectorize


@vectorize(nopython=True)
def ufunc_kernel(x, y):
    return x + y


def get_device():
    device = None
    try:
        device = dpctl.select_gpu_device()
    except:
        try:
            device = dpctl.select_cpu_device()
        except:
            raise RuntimeError("No device found")
    return device


def test_njit():
    N = 10
    dtype = np.float64

    A = np.arange(N, dtype=dtype)
    B = np.arange(N, dtype=dtype) * 10

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        C = ufunc_kernel(A, B)

    print(C)

    print("Done...")


@vectorize([float64(float64, float64)], target="dpex")
def vector_add(a, b):
    return a + b


def test_vectorize():
    A = np.arange(10, dtype=np.float64).reshape((5, 2))
    B = np.arange(10, dtype=np.float64).reshape((5, 2))

    device = dpctl.select_default_device()
    with dpctl.device_context(device):
        C = vector_add(A, B)

    print(C)


if __name__ == "__main__":
    test_njit()
    test_vectorize()
