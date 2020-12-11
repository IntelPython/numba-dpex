import numpy as np

import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


@dppy.kernel(access_types={"read_only": ['a', 'b'], "write_only": ['c'], "read_write": []})
def sum_with_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

@dppy.kernel
def sum_without_accessor(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]

def call_kernel(global_size, local_size,
                A, B, C, func):
        func[global_size, dppy.DEFAULT_LOCAL_SIZE](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size

A = np.array(np.random.random(N), dtype=np.float32)
B = np.array(np.random.random(N), dtype=np.float32)
D = A + B


@unittest.skipUnless(dpctl.has_cpu_queues(), 'test only on CPU system')
class TestDPPYArgAccessorCPU(unittest.TestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            call_kernel(global_size, local_size,
                        A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            call_kernel(global_size, local_size,
                        A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPYArgAccessorGPU(unittest.TestCase):
    def test_arg_with_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            call_kernel(global_size, local_size,
                        A, B, C, sum_with_accessor)
        self.assertTrue(np.all(D == C))

    def test_arg_without_accessor(self):
        C = np.ones_like(A)
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            call_kernel(global_size, local_size,
                        A, B, C, sum_without_accessor)
        self.assertTrue(np.all(D == C))


if __name__ == '__main__':
    unittest.main()
