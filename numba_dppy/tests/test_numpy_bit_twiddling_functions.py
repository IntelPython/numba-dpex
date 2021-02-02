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


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestNumpy_bit_twiddling_functions(unittest.TestCase):
    def test_bitwise_and(self):
        @njit
        def f(a, b):
            c = np.bitwise_and(a, b)
            return c

        a = np.array([2, 5, 255])
        b = np.array([3, 14, 16])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.bitwise_and(a, b)
        self.assertTrue(np.all(c == d))

    def test_bitwise_or(self):
        @njit
        def f(a, b):
            c = np.bitwise_or(a, b)
            return c

        a = np.array([2, 5, 255])
        b = np.array([4, 4, 4])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.bitwise_or(a, b)
        self.assertTrue(np.all(c == d))

    def test_bitwise_xor(self):
        @njit
        def f(a, b):
            c = np.bitwise_xor(a, b)
            return c

        a = np.array([2, 5, 255])
        b = np.array([4, 4, 4])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.bitwise_xor(a, b)
        self.assertTrue(np.all(c == d))

    def test_bitwise_not(self):
        @njit
        def f(a):
            c = np.bitwise_not(a)
            return c

        a = np.array([2, 5, 255])

        with dpctl.device_context("opencl:gpu"):
            c = f(a)

        d = np.bitwise_not(a)
        self.assertTrue(np.all(c == d))

    def test_invert(self):
        @njit
        def f(a):
            c = np.invert(a)
            return c

        a = np.array([2, 5, 255])

        with dpctl.device_context("opencl:gpu"):
            c = f(a)

        d = np.invert(a)
        self.assertTrue(np.all(c == d))

    def test_left_shift(self):
        @njit
        def f(a, b):
            c = np.left_shift(a, b)
            return c

        a = np.array([2, 3, 4])
        b = np.array([1, 2, 3])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.left_shift(a, b)
        self.assertTrue(np.all(c == d))

    def test_right_shift(self):
        @njit
        def f(a, b):
            c = np.right_shift(a, b)
            return c

        a = np.array([2, 3, 4])
        b = np.array([1, 2, 3])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.right_shift(a, b)
        self.assertTrue(np.all(c == d))


if __name__ == "__main__":
    unittest.main()
