import numpy as np
from numba import njit
import dpctl
import unittest


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestNumpy_floating_functions(unittest.TestCase):
    def test_isfinite(self):
        @njit
        def f(a):
            c = np.isfinite(a)
            return c

        test_arr = [np.log(-1.0), 1.0, np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.isfinite(input_arr)
        self.assertTrue(np.all(c == d))

    def test_isinf(self):
        @njit
        def f(a):
            c = np.isinf(a)
            return c

        test_arr = [np.log(-1.0), 1.0, np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.isinf(input_arr)
        self.assertTrue(np.all(c == d))

    def test_isnan(self):
        @njit
        def f(a):
            c = np.isnan(a)
            return c

        test_arr = [np.log(-1.0), 1.0, np.log(0)]
        input_arr = np.asarray(test_arr, dtype=np.float32)

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.isnan(input_arr)
        self.assertTrue(np.all(c == d))

    def test_floor(self):
        @njit
        def f(a):
            c = np.floor(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.floor(input_arr)
        self.assertTrue(np.all(c == d))

    def test_ceil(self):
        @njit
        def f(a):
            c = np.ceil(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.ceil(input_arr)
        self.assertTrue(np.all(c == d))

    def test_trunc(self):
        @njit
        def f(a):
            c = np.trunc(a)
            return c

        input_arr = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])

        with dpctl.device_context("opencl:gpu"):
            c = f(input_arr)

        d = np.trunc(input_arr)
        self.assertTrue(np.all(c == d))


if __name__ == "__main__":
    unittest.main()
