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
import numba
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
from numba_dppy import config as dppy_config
from numba_dppy.testing import unittest
from numba.tests.support import captured_stdout
import dpctl
from numba_dppy.context import device_context


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestOffloadDiagnostics(unittest.TestCase):
    def test_parfor(self):
        def prange_func():
            n = 10
            a = np.ones((n), dtype=np.float64)
            b = np.ones((n), dtype=np.float64)
            c = np.ones((n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        with device_context("opencl:gpu"):
            dppy_config.OFFLOAD_DIAGNOSTICS = 1
            jitted = njit(parallel=True)(prange_func)

            with captured_stdout() as got:
                jitted()

            dppy_config.OFFLOAD_DIAGNOSTICS = 0
            self.assertTrue("Auto-offloading" in got.getvalue())
            self.assertTrue("Device -" in got.getvalue())

    def test_kernel(self):
        @dppy.kernel
        def parallel_sum(a, b, c):
            i = dppy.get_global_id(0)
            c[i] = a[i] + b[i]

        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        with device_context("opencl:gpu"):
            dppy_config.OFFLOAD_DIAGNOSTICS = 1

            with captured_stdout() as got:
                parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

            dppy_config.OFFLOAD_DIAGNOSTICS = 0
            self.assertTrue("Auto-offloading" in got.getvalue())
            self.assertTrue("Device -" in got.getvalue())


if __name__ == "__main__":
    unittest.main()
