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

import numba_dppy as dppy
import unittest
import dpctl
from numba_dppy.context_manager import offload_to_sycl_device
from . import _helper


@unittest.skipUnless(_helper.has_gpu_queues(), "test only on GPU system")
class TestDPPYFunc(unittest.TestCase):
    N = 257

    def test_dppy_func_device_array(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu"):
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

        self.assertTrue(np.all(b == 2))

    def test_dppy_func_ndarray(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        @dppy.kernel
        def h(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu"):
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 2))

            h[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 3))


if __name__ == "__main__":
    unittest.main()
