import sys
import numpy as np
import numba_dppy, numba_dppy as dppy
import dpctl
import unittest


def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


class TestCaching(unittest.TestCase):
    def test_caching_kernel(self):
        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            func = dppy.kernel(data_parallel_sum)
            caching_kernel = func[global_size, dppy.DEFAULT_LOCAL_SIZE].specialize(
                a, b, c
            )

            for i in range(10):
                cached_kernel = func[global_size, dppy.DEFAULT_LOCAL_SIZE].specialize(
                    a, b, c
                )
                self.assertIs(caching_kernel, cached_kernel)


if __name__ == "__main__":
    unittest.main()
