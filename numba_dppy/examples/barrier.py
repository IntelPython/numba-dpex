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

import unittest
from numba import float32
import numba_dppy, numba_dppy as dppy
import dpctl


def no_arg_barrier_support():
    # @dppy.kernel("void(float32[::1])")
    @dppy.kernel
    def twice(A):
        i = dppy.get_global_id(0)
        d = A[i]
        # no argument defaults to global mem fence
        dppy.barrier()
        A[i] = d * 2

    N = 10
    arr = np.arange(N).astype(np.float32)
    print(arr)

    with dpctl.device_context("opencl:gpu") as gpu_queue:
        twice[N, dppy.DEFAULT_LOCAL_SIZE](arr)

    # there arr should be original arr * 2, i.e. [0, 2, 4, 6, ...]
    print(arr)


def local_memory():
    blocksize = 10

    # @dppy.kernel("void(float32[::1])")
    @dppy.kernel
    def reverse_array(A):
        lm = dppy.local.static_alloc(shape=10, dtype=float32)
        i = dppy.get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    arr = np.arange(blocksize).astype(np.float32)
    print(arr)

    with dpctl.device_context("opencl:gpu") as gpu_queue:
        reverse_array[blocksize, dppy.DEFAULT_LOCAL_SIZE](arr)

    # there arr should be orig[::-1] + orig, i.e. [9, 9, 9, ...]
    print(arr)


def main():
    no_arg_barrier_support()
    local_memory()


if __name__ == "__main__":
    main()
