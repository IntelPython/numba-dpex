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

import unittest

import dpctl
import numpy as np

import numba_dppy as dppy

from . import _helper


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
d = a + b


@unittest.skipUnless(_helper.has_cpu_queues(), "test only on CPU system")
class TestDPPYDeviceArrayArgsGPU(unittest.TestCase):
    def test_device_array_args_cpu(self):
        c = np.ones_like(a)

        with dppy.offload_to_sycl_device("opencl:cpu"):
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

            self.assertTrue(np.all(c == d))


@unittest.skipUnless(_helper.has_gpu_queues(), "test only on GPU system")
class TestDPPYDeviceArrayArgsCPU(unittest.TestCase):
    def test_device_array_args_gpu(self):
        c = np.ones_like(a)

        with dppy.offload_to_sycl_device("opencl:gpu"):
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

        self.assertTrue(np.all(c == d))


if __name__ == "__main__":
    unittest.main()
