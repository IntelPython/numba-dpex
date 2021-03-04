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

import sys
import numpy as np
import numba
import dpctl
from numba_dppy.context import device_context
from numba import njit, prange
import numba_dppy
import unittest
from numba_dppy.testing import expectedFailureIf
from numba.tests.support import captured_stdout


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestPrange(unittest.TestCase):
    def test_one_prange(self):
        @njit
        def f(a, b):
            for i in prange(4):
                b[i, 0] = a[i, 0] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        with device_context("opencl:gpu"):
            f(a, b)

        for i in range(4):
            self.assertTrue(b[i, 0] == a[i, 0] * 10)

    def test_nested_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                for j in prange(n):
                    b[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        with device_context("opencl:gpu"):
            f(a, b)

        self.assertTrue(np.all(b == 10))

    def test_multiple_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    b[i, j] = a[i, j] * val

            for i in prange(m):
                for j in prange(n):
                    a[i, j] = a[i, j] * 10

        m = 8
        n = 8
        a = np.ones((m, n))
        b = np.ones((m, n))

        with device_context("opencl:gpu"):
            f(a, b)

        self.assertTrue(np.all(b == 10))
        self.assertTrue(np.all(a == 10))

    def test_three_prange(self):
        @njit
        def f(a, b):
            # dimensions must be provided as scalar
            m, n, o = a.shape
            for i in prange(m):
                val = 10
                for j in prange(n):
                    constant = 2
                    for k in prange(o):
                        b[i, j, k] = a[i, j, k] * (val + constant)

        m = 8
        n = 8
        o = 8
        a = np.ones((m, n, o))
        b = np.ones((m, n, o))

        with device_context("opencl:gpu"):
            f(a, b)

        self.assertTrue(np.all(b == 12))

    @unittest.skip("numba-dppy issue 110")
    def test_two_consequent_prange(self):
        def prange_example():
            n = 10
            a = np.ones((n), dtype=np.float64)
            b = np.ones((n), dtype=np.float64)
            c = np.ones((n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        old_debug = numba_dppy.compiler.DEBUG
        numba_dppy.compiler.DEBUG = 1

        jitted = njit(prange_example)

        with captured_stdout() as stdout, device_context("opencl:gpu"):
            jitted_res = jitted()

        res = prange_example()

        numba_dppy.compiler.DEBUG = old_debug

        self.assertEqual(
            stdout.getvalue().count("Parfor lowered on DPPY-device"),
            2,
            stdout.getvalue(),
        )
        self.assertEqual(
            stdout.getvalue().count("Failed to lower parfor on DPPY-device"),
            0,
            stdout.getvalue(),
        )
        np.testing.assert_equal(res, jitted_res)

    @unittest.skip("NRT required but not enabled")
    def test_2d_arrays(self):
        def prange_example():
            n = 10
            a = np.ones((n, n), dtype=np.float64)
            b = np.ones((n, n), dtype=np.float64)
            c = np.ones((n, n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        old_debug = numba_dppy.compiler.DEBUG
        numba_dppy.compiler.DEBUG = 1

        jitted = njit(prange_example)

        with captured_stdout() as stdout, device_context("opencl:gpu"):
            jitted_res = jitted()

        res = prange_example()

        numba_dppy.compiler.DEBUG = old_debug

        self.assertEqual(
            stdout.getvalue().count("Parfor lowered on DPPY-device"),
            2,
            stdout.getvalue(),
        )
        self.assertEqual(
            stdout.getvalue().count("Failed to lower parfor on DPPY-device"),
            0,
            stdout.getvalue(),
        )
        np.testing.assert_equal(res, jitted_res)


if __name__ == "__main__":
    unittest.main()
