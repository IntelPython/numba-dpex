import numpy as np

import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPYFunc(unittest.TestCase):
    N = 257

    def test_dppy_func_device_array(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)


        self.assertTrue(np.all(b == 2))

    def test_dppy_func_ndarray(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        @dppy.kernel
        def h(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 2))

            h[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            self.assertTrue(np.all(b == 3))


if __name__ == '__main__':
    unittest.main()
