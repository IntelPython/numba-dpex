import numpy as np

import numba
import unittest
from numba.tests.support import captured_stderr
import dpctl
import warnings


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestDPPYFallback(unittest.TestCase):
    def test_dppy_fallback_inner_call(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        with warnings.catch_warnings(record=True) as w, dpctl.device_context("opencl:gpu"):
            dppy = numba.njit(inner_call_fallback)
            dppy_result = dppy()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(dppy_result, ref_result)
        assert 'Failed to lower parfor on DPPY-device' in str(w[-1].message)

    def test_dppy_fallback_reductions(self):
        def reduction(a):
            b = 1
            for i in numba.prange(len(a)):
                b += a[i]
            return b

        a = np.ones(10)
        with warnings.catch_warnings(record=True) as w, dpctl.device_context("opencl:gpu"):
            dppy = numba.njit(reduction)
            dppy_result = dppy(a)

        ref_result = reduction(a)

        np.testing.assert_array_equal(dppy_result, ref_result)
        assert 'Failed to lower parfor on DPPY-device' in str(w[-1].message)


if __name__ == "__main__":
    unittest.main()
