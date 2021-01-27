import numpy as np
import math
import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


@dppy.kernel
def reduction_kernel(A, R, stride):
    i = dppy.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestDPPYSumReduction(unittest.TestCase):
    def test_sum_reduction(self):
        # This test will only work for even case
        N = 1024
        self.assertTrue(N % 2 == 0)

        A = np.array(np.random.random(N), dtype=np.float32)
        A_copy = A.copy()
        # at max we will require half the size of A to store sum
        R = np.array(np.random.random(math.ceil(N / 2)), dtype=np.float32)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            total = N

            while total > 1:
                # call kernel
                global_size = total // 2
                reduction_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](
                    A, R, global_size
                )
                total = total // 2

            result = A_copy.sum()
            max_abs_err = result - R[0]
            self.assertTrue(max_abs_err < 1e-4)


if __name__ == "__main__":
    unittest.main()
