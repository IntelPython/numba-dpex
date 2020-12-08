from __future__ import print_function, division, absolute_import

import numpy as np

import numba
import numba_dppy, numba_dppy as dppy
from numba_dppy.testing import unittest
from numba_dppy.testing import DPPYTestCase
from numba.tests.support import captured_stderr
import dpctl
import sys
import io


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPYFallback(DPPYTestCase):
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

        with captured_stderr() as msg:
            dppy = numba.njit(parallel={'offload':True})(inner_call_fallback)
            dppy_result = dppy()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(dppy_result, ref_result)
        self.assertTrue('Failed to lower parfor on DPPY-device' in msg.getvalue())

    def test_dppy_fallback_reductions(self):
        def reduction(a):
            return np.amax(a)

        a = np.ones(10)
        with captured_stderr() as msg:
            dppy = numba.njit(parallel={'offload':True})(reduction)
            dppy_result = dppy(a)

        ref_result = reduction(a)

        np.testing.assert_array_equal(dppy_result, ref_result)
        self.assertTrue('Failed to lower parfor on DPPY-device' in msg.getvalue())


if __name__ == '__main__':
    unittest.main()
