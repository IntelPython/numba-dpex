#! /usr/bin/env python
import sys
import numpy as np
import numba_dppy, numba_dppy as dppy
import dpctl
import unittest

@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
d = a + b


@unittest.skipUnless(dpctl.has_cpu_queues(), 'test only on CPU system')
class TestDPPYDeviceArrayArgsGPU(unittest.TestCase):
    def test_device_array_args_cpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:cpu") as cpu_queue:
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

            self.assertTrue(np.all(c == d))


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestDPPYDeviceArrayArgsCPU(unittest.TestCase):
    def test_device_array_args_gpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

        self.assertTrue(np.all(c == d))


if __name__ == '__main__':
    unittest.main()
