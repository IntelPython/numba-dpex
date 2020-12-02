from __future__ import print_function, division, absolute_import

import numpy as np

import numba
import numba_dppy, numba_dppy as dppl
from numba_dppy.testing import unittest
from numba_dppy.testing import DPPLTestCase
from numba.tests.support import captured_stderr
import dpctl
import sys
import io


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPLFallback(DPPLTestCase):
    def test_dppl_fallback_true(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        with captured_stderr() as msg_fallback_true:
            dppl = numba.njit(parallel={'offload':True, 'fallback':True})(inner_call_fallback)
            dppl_fallback_true = dppl()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(dppl_fallback_true, ref_result)
        self.assertTrue('Failed to lower parfor on DPPL-device' in msg_fallback_true.getvalue())

    @unittest.expectedFailure
    def test_dppl_fallback_false(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        dppl = numba.njit(parallel={'offload':True, 'fallback':False})(inner_call_fallback)
        dppl_fallback_false = dppl()

        ref_result = inner_call_fallback()

        not np.testing.assert_array_equal(dppl_fallback_false, ref_result)

    @unittest.expectedFailure
    def test_dppl_fallback_non(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        dppl = numba.njit(parallel={'offload':True})(inner_call_fallback)
        dppl_fallback_non = dppl()

        ref_result = inner_call_fallback()

        not np.testing.assert_array_equal(dppl_fallback_non, ref_result)


if __name__ == '__main__':
    unittest.main()
