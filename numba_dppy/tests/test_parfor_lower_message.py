import numpy as np
import numba
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
from numba_dppy.testing import unittest, DPPYTestCase
from numba.tests.support import captured_stdout
import dpctl


def prange_example():
    n = 10
    a = np.ones((n), dtype=np.float64)
    b = np.ones((n), dtype=np.float64)
    c = np.ones((n), dtype=np.float64)
    for i in prange(n//2):
        a[i] = b[i] + c[i]

    return a


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestParforMessage(DPPYTestCase):
    def test_parfor_message(self):
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            numba_dppy.compiler.DEBUG = 1
            jitted = njit(parallel={"offload": True})(prange_example)

            with captured_stdout() as got:
                jitted()

            numba_dppy.compiler.DEBUG = 0
            self.assertTrue("Parfor lowered on DPPY-device" in got.getvalue())


if __name__ == '__main__':
    unittest.main()
