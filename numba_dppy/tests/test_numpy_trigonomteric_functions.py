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
from numba import njit
import dpctl
import unittest
from . import skip_tests


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestNumpy_math_functions(unittest.TestCase):

    N = 10
    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)

    def test_sin(self):
        @njit
        def f(a):
            c = np.sin(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.sin(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_cos(self):
        @njit
        def f(a):
            c = np.cos(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.cos(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_tan(self):
        @njit
        def f(a):
            c = np.tan(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.tan(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arcsin(self):
        @njit
        def f(a):
            c = np.arcsin(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.arcsin(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arccos(self):
        @njit
        def f(a):
            c = np.arccos(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.arccos(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arctan(self):
        @njit
        def f(a):
            c = np.arctan(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.arctan(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arctan2(self):
        @njit
        def f(a, b):
            c = np.arctan2(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.arctan2(self.a, self.b)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_sinh(self):
        @njit
        def f(a):
            c = np.sinh(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.sinh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_cosh(self):
        @njit
        def f(a):
            c = np.cosh(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.cosh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_tanh(self):
        @njit
        def f(a):
            c = np.tanh(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.tanh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arcsinh(self):
        @njit
        def f(a):
            c = np.arcsinh(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.arcsinh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    @unittest.skipIf(skip_tests.is_gen12("opencl:gpu"), "Gen12 not supported")
    def test_arccosh(self):
        @njit
        def f(a):
            c = np.arccosh(a)
            return c

        input_arr = np.random.randint(1, self.N, size=(self.N))

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.arccosh(input_arr)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_arctanh(self):
        @njit
        def f(a):
            c = np.arctanh(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.arctanh(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_deg2rad(self):
        @njit
        def f(a):
            c = np.deg2rad(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.deg2rad(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)

    def test_rad2deg(self):
        @njit
        def f(a):
            c = np.rad2deg(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.rad2deg(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_degrees(self):
        @njit
        def f(a):
            c = np.degrees(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.degrees(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-2)

    def test_radians(self):
        @njit
        def f(a):
            c = np.radians(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a)

        d = np.radians(self.a)
        max_abs_err = c.sum() - d.sum()
        self.assertTrue(max_abs_err < 1e-5)


if __name__ == "__main__":
    unittest.main()
