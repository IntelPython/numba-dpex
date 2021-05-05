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
import numba_dppy as dppy
import unittest
from numba.tests.support import captured_stdout
import dpctl
from . import _helper


def prange_example():
    n = 10
    a = np.ones((n), dtype=np.float64)
    b = np.ones((n), dtype=np.float64)
    c = np.ones((n), dtype=np.float64)
    for i in prange(n // 2):
        a[i] = b[i] + c[i]

    return a


@unittest.skipUnless(_helper.has_gpu_queues(), "test only on GPU system")
class TestParforMessage(unittest.TestCase):
    def test_parfor_message(self):
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            dppy.compiler.DEBUG = 1
            jitted = njit(prange_example)

            with captured_stdout() as got:
                jitted()

            dppy.compiler.DEBUG = 0
            self.assertTrue("Parfor lowered to specified SYCL device" in got.getvalue())


if __name__ == "__main__":
    unittest.main()
