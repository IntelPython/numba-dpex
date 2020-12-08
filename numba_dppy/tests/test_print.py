#! /usr/bin/env python
from timeit import default_timer as time

import sys
import numpy as np
from numba import njit, prange
import numba_dppy, numba_dppy as dppy
import unittest

import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), 'test only on GPU system')
class TestPrint(unittest.TestCase):
    def test_print_dppy_kernel(self):
        @dppy.func
        def g(a):
            print("value of a:", a)
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])
            print("value of b at:", i, "is", b[i])

        N = 10

        a = np.ones(N)
        b = np.ones(N)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            f[N, dppy.DEFAULT_LOCAL_SIZE](a, b)


if __name__ == '__main__':
    unittest.main()
