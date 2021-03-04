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

import numpy as np
from numba import njit, vectorize
import dpctl
from numba_dppy.context import device_context
import unittest


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestVectorize(unittest.TestCase):
    def test_vectorize(self):
        @vectorize(nopython=True)
        def axy(a, x, y):
            return a * x + y

        @njit
        def f(a0, a1):
            return np.cos(axy(a0, np.sin(a1) - 1.0, 1.0))

        def f_np(a0, a1):
            sin_res = np.sin(a1)
            res = []
            for i in range(len(a0)):
                res.append(axy(a0[i], sin_res[i] - 1.0, 1.0))
            return np.cos(np.array(res))

        A = np.random.random(10)
        B = np.random.random(10)

        with device_context("opencl:gpu"):
            expected = f(A, B)

        actual = f_np(A, B)

        max_abs_err = expected.sum() - actual.sum()
        self.assertTrue(max_abs_err < 1e-5)


if __name__ == "__main__":
    unittest.main()
