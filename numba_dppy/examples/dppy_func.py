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

import sys
import numpy as np
import numba_dppy, numba_dppy as dppy
import math

import dpctl


@dppy.func
def g(a):
    return a + 1


@dppy.kernel
def f(a, b):
    i = dppy.get_global_id(0)
    b[i] = g(a[i])


def driver(a, b, N):
    print(b)
    print("--------")
    f[N, dppy.DEFAULT_LOCAL_SIZE](a, b)
    print(b)


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, N)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            driver(a, b, N)
    else:
        print("No device found")


if __name__ == "__main__":
    main()
