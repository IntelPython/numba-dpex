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

import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


@dppy.kernel(
    access_types={"read_only": ["a", "b"], "write_only": ["c"], "read_write": []}
)
def sum_with_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


@dppy.kernel
def sum_without_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def call_kernel(global_size, local_size, A, B, C, func):
    func[global_size, dppy.DEFAULT_LOCAL_SIZE](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size

A = np.array(np.random.random(N), dtype=np.float32)
B = np.array(np.random.random(N), dtype=np.float32)
D = A + B


@unittest.skipUnless(dpctl.has_cpu_queues(), "test only on OpenCL CPU system")
class TestDPPYArgAccessorCPU(unittest.TestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on OpenCL GPU system")
class TestDPPYArgAccessorOCLGPU(unittest.TestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


@unittest.skipUnless(
    dpctl.has_gpu_queues(dpctl.backend_type.level_zero),
    "test only on Level Zero GPU system",
)
class TestDPPYArgAccessorL0GPU(unittest.TestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("level0:gpu") as gpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("level0:gpu") as gpu_queue:
            call_kernel(global_size, local_size, A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


if __name__ == "__main__":
    unittest.main()
