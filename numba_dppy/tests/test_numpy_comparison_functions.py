#! /usr/bin/env python
import numpy as np
from numba import njit
import dpctl
import unittest


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestNumpy_comparison_functions(unittest.TestCase):
    a = np.array([4, 5, 6])
    b = np.array([2, 6, 6])

    def test_greater(self):
        @njit
        def f(a, b):
            c = np.greater(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.greater(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_greater_equal(self):
        @njit
        def f(a, b):
            c = np.greater_equal(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.greater_equal(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_less(self):
        @njit
        def f(a, b):
            c = np.less(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.less(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_less_equal(self):
        @njit
        def f(a, b):
            c = np.less_equal(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.less_equal(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_not_equal(self):
        @njit
        def f(a, b):
            c = np.not_equal(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.not_equal(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_equal(self):
        @njit
        def f(a, b):
            c = np.equal(a, b)
            return c

        with dpctl.device_context("opencl:gpu"):
            c = f(self.a, self.b)

        d = np.equal(self.a, self.b)
        self.assertTrue(np.all(c == d))

    def test_logical_and(self):
        @njit
        def f(a, b):
            c = np.logical_and(a, b)
            return c

        a = np.array([True, True, False])
        b = np.array([True, False, False])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.logical_and(a, b)
        self.assertTrue(np.all(c == d))

    def test_logical_or(self):
        @njit
        def f(a, b):
            c = np.logical_or(a, b)
            return c

        a = np.array([True, True, False])
        b = np.array([True, False, False])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.logical_or(a, b)
        self.assertTrue(np.all(c == d))

    def test_logical_xor(self):
        @njit
        def f(a, b):
            c = np.logical_xor(a, b)
            return c

        a = np.array([True, True, False])
        b = np.array([True, False, False])

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.logical_xor(a, b)
        self.assertTrue(np.all(c == d))

    def test_logical_not(self):
        @njit
        def f(a):
            c = np.logical_not(a)
            return c

        a = np.array([True, True, False])

        with dpctl.device_context("opencl:gpu"):
            c = f(a)

        d = np.logical_not(a)
        self.assertTrue(np.all(c == d))

    def test_maximum(self):
        @njit
        def f(a, b):
            c = np.maximum(a, b)
            return c

        a = np.array([5, 6, 7, np.nan], dtype=np.float32)
        b = np.array([5, 7, 6, 100], dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.maximum(a, b)
        np.testing.assert_equal(c, d)

    def test_minimum(self):
        @njit
        def f(a, b):
            c = np.minimum(a, b)
            return c

        a = np.array([5, 6, 7, np.nan], dtype=np.float32)
        b = np.array([5, 7, 6, 100], dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.minimum(a, b)
        np.testing.assert_equal(c, d)

    def test_fmax(self):
        @njit
        def f(a, b):
            c = np.fmax(a, b)
            return c

        a = np.array([5, 6, 7, np.nan], dtype=np.float32)
        b = np.array([5, 7, 6, 100], dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.fmax(a, b)
        np.testing.assert_equal(c, d)

    def test_fmin(self):
        @njit
        def f(a, b):
            c = np.fmin(a, b)
            return c

        a = np.array([5, 6, 7, np.nan], dtype=np.float32)
        b = np.array([5, 7, 6, 100], dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(a, b)

        d = np.fmin(a, b)
        np.testing.assert_equal(c, d)


if __name__ == "__main__":
    unittest.main()
