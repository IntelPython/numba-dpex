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
import numba_dppy
from numba_dppy.context import device_context
import numpy as np
from numba import njit
from numba.core import errors
from numba.tests.support import captured_stdout


class TestWithDPPYContext(unittest.TestCase):
    @unittest.skipIf(not dpctl.has_gpu_queues(), "No GPU platforms available")
    def test_with_dppy_context_gpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        numba_dppy.compiler.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_gpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_gpu_message:
            with device_context("opencl:gpu"):
                func(got_gpu)

        numba_dppy.compiler.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_gpu)
        self.assertIn("Parfor lowered on DPPY-device", got_gpu_message.getvalue())

    @unittest.skipIf(not dpctl.has_cpu_queues(), "No CPU platforms available")
    def test_with_dppy_context_cpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        numba_dppy.compiler.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_cpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_cpu_message:
            with device_context("opencl:cpu"):
                func(got_cpu)

        numba_dppy.compiler.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_cpu)
        self.assertIn("Parfor lowered on DPPY-device", got_cpu_message.getvalue())

    @unittest.skipIf(not dpctl.has_gpu_queues(), "No GPU platforms available")
    def test_with_dppy_context_target(self):
        @njit(target="cpu")
        def nested_func_target(a, b):
            np.sin(a, b)

        @njit(target="gpu")
        def func_target(b):
            a = np.ones((64), dtype=np.float64)
            nested_func_target(a, b)

        @njit
        def func_no_target(b):
            a = np.ones((64), dtype=np.float64)
            nested_func_target(a, b)

        @njit(parallel=False)
        def func_no_parallel(b):
            a = np.ones((64), dtype=np.float64)
            return a

        a = np.ones((64), dtype=np.float64)
        b = np.ones((64), dtype=np.float64)

        with self.assertRaises(errors.UnsupportedError) as raises_1:
            with device_context("opencl:gpu"):
                nested_func_target(a, b)

        with self.assertRaises(errors.UnsupportedError) as raises_2:
            with device_context("opencl:gpu"):
                func_target(a)

        with self.assertRaises(errors.UnsupportedError) as raises_3:
            with device_context("opencl:gpu"):
                func_no_target(a)

        with self.assertRaises(errors.UnsupportedError) as raises_4:
            with device_context("opencl:gpu"):
                func_no_parallel(a)

        msg_1 = "Can't use 'with' context with explicitly specified target"
        msg_2 = "Can't use 'with' context with parallel option"
        self.assertTrue(msg_1 in str(raises_1.exception))
        self.assertTrue(msg_1 in str(raises_2.exception))
        self.assertTrue(msg_1 in str(raises_3.exception))
        self.assertTrue(msg_2 in str(raises_4.exception))


if __name__ == "__main__":
    unittest.main()
