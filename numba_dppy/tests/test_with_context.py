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
from numba import njit
import numba_dppy
from numba_dppy import config
import unittest
from numba.core import errors
from numba.tests.support import captured_stdout
from . import _helper
import dpctl
from numba_dppy.context_manager import offload_to_sycl_device


class TestWithDPPYContext(unittest.TestCase):
    @unittest.skipIf(not _helper.has_gpu_queues(), "No GPU platforms available")
    def test_with_dppy_context_gpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        config.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_gpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_gpu_message:
            device = dpctl.SyclDevice("opencl:gpu")
            with offload_to_sycl_device(device):
                func(got_gpu)

        config.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_gpu)
        self.assertTrue("Parfor offloaded to opencl:gpu" in got_gpu_message.getvalue())

    @unittest.skipIf(not _helper.has_cpu_queues(), "No CPU platforms available")
    def test_with_dppy_context_cpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        config.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_cpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_cpu_message:
            device = dpctl.SyclDevice("opencl:cpu")
            with offload_to_sycl_device(device):
                func(got_cpu)

        config.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_cpu)
        self.assertTrue("Parfor offloaded to opencl:cpu" in got_cpu_message.getvalue())


if __name__ == "__main__":
    unittest.main()
