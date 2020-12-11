import numpy as np

import numba
import numba_dppy
from numba_dppy.testing import unittest
from numba.tests.support import captured_stderr
import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPYFallback(unittest.TestCase):
    def test_dppy_fallback_true(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        numba_dppy.compiler.DEBUG = 1
        with captured_stderr() as msg_fallback_true:
            with dpctl.device_context("opencl:gpu") as gpu_queue:
                dppy = numba.njit(parallel=True)(inner_call_fallback)
                dppy_fallback_true = dppy()

        ref_result = inner_call_fallback()
        numba_dppy.compiler.DEBUG = 0

        np.testing.assert_array_equal(dppy_fallback_true, ref_result)
        self.assertTrue('Failed to lower parfor on DPPY-device' in msg_fallback_true.getvalue())

    @unittest.expectedFailure
    def test_dppy_fallback_false(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        try:
            numba_dppy.compiler.DEBUG = 1
            numba_dppy.config.FALLBACK_ON_CPU  = 0
            with captured_stderr() as msg_fallback_true:
                with dpctl.device_context("opencl:gpu") as gpu_queue:
                    dppy = numba.njit(parallel=True)(inner_call_fallback)
                    dppy_fallback_false = dppy()

        finally:
            ref_result = inner_call_fallback()
            numba_dppy.config.FALLBACK_ON_CPU  = 1
            numba_dppy.compiler.DEBUG = 0

            not np.testing.assert_array_equal(dppy_fallback_false, ref_result)
            not self.assertTrue('Failed to lower parfor on DPPY-device' in msg_fallback_true.getvalue())


if __name__ == '__main__':
    unittest.main()
