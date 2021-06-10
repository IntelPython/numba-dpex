#! /usr/bin/env python
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from timeit import default_timer as time
import numpy as np
import numba_dppy as dppy
import dpctl


@dppy.kernel
def dppy_gemm(a, b, c):
    """
    A basic DGEMM implemented as a ``kernel`` function.
    """
    i = dppy.get_global_id(0)
    j = dppy.get_global_id(1)
    if i >= c.shape[0] or j >= c.shape[1]:
        return
    c[i, j] = 0
    for k in range(c.shape[0]):
        c[i, j] += a[i, k] * b[k, j]


# Array dimensions
X = 1024
Y = 16
global_size = X, X

griddim = X, X
blockdim = Y, Y


def driver(a, b, c):
    # Invoke the kernel
    dppy_gemm[griddim, blockdim](a, b, c)


def main():
    a = np.arange(X * X, dtype=np.float32).reshape(X, X)
    b = np.array(np.random.random(X * X), dtype=np.float32).reshape(X, X)
    c = np.ones_like(a).reshape(X, X)

    try:
        device = dpctl.select_default_device()
        print("Running on the following SYCL GPU device")
        device.print_device_info()
        with dpctl.device_context(device):
            driver(a, b, c)
    except ValueError:
        print("No SYCL GPU device found")

    # Host compute using standard NumPy
    Amat = np.matrix(a)
    Bmat = np.matrix(b)
    Cans = Amat * Bmat

    # Check result
    assert np.allclose(c, Cans)

    print("Done...")


if __name__ == "__main__":
    main()
