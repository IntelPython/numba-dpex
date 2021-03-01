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
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
import unittest

import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestPrint(unittest.TestCase):
    def test_print_dppy_kernel(self):
        @dppy.func
        def g(a):
            print("value of a:", a)
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])
            print("value of b at:", i, "is", b[i])

        N = 10

        a = np.ones(N)
        b = np.ones(N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[N, dppy.DEFAULT_LOCAL_SIZE](a, b)


if __name__ == "__main__":
    unittest.main()
