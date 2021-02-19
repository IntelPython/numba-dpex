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

from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
import numba_dppy, numba_dppy as dppy
import dpctl


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


N = 50 * 32
global_size = (N,)


def main():
    if dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            print("-----Running in CPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
            print("after  C: ", c)
    else:
        print("CPU device not found")

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            print("-----Running in GPU-----")
            a = np.array(np.random.random(N), dtype=np.float32)
            b = np.array(np.random.random(N), dtype=np.float32)
            c = np.ones_like(a)
            print("before A: ", a)
            print("before B: ", b)
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
            print("after  C: ", c)
    else:
        print("GPU device not found")
        exit()

    print("Done...")


if __name__ == "__main__":
    main()
