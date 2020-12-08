import numpy as np
import numba
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
from numba_dppy import config as dppy_config
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


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestOffloadDiagnostics(DPPYTestCase):
    def test_parfor(self):
        with dpctl.device_context("opencl:gpu"):
            dppy_config.OFFLOAD_DIAGNOSTICS = 1
            jitted = njit(parallel={"offload": True})(prange_example)

            with captured_stdout() as got:
                jitted()

            dppy_config.OFFLOAD_DIAGNOSTICS = 0
            self.assertTrue("Auto-offloading" in got.getvalue())
            self.assertTrue("Device -" in got.getvalue())

    def test_kernel(self):
        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            dppy_config.OFFLOAD_DIAGNOSTICS = 1

            with captured_stdout() as got:
                driver(a, b, c, global_size)

            dppy_config.OFFLOAD_DIAGNOSTICS = 0
            self.assertTrue("Auto-offloading" in got.getvalue())
            self.assertTrue("Device -" in got.getvalue())


if __name__ == '__main__':
    unittest.main()
