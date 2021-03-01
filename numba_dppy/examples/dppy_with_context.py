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

import numpy as np
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
import dpctl


@njit
def add_two_arrays(b, c):
    a = np.empty_like(b)
    for i in prange(len(b)):
        a[i] = b[i] + c[i]

    return a


def main():
    N = 10
    b = np.ones(N)
    c = np.ones(N)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu"):
            gpu_result = add_two_arrays(b, c)
        print("GPU device found. Result on GPU:", gpu_result)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu"):
            cpu_result = add_two_arrays(b, c)
        print("CPU device found. Result on CPU:", cpu_result)
    else:
        print("No device found")


if __name__ == "__main__":
    main()
