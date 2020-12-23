import numpy as np

import unittest
from numba import float32
import numba_dppy, numba_dppy as dppy
import dpctl


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestBarrier(unittest.TestCase):
    def test_proper_lowering(self):
        # @dppy.kernel("void(float32[::1])")
        @dppy.kernel
        def twice(A):
            i = dppy.get_global_id(0)
            d = A[i]
            dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)  # local mem fence
            A[i] = d * 2

        N = 256
        arr = np.random.random(N).astype(np.float32)
        orig = arr.copy()

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            twice[N, N // 2](arr)

        # The computation is correct?
        np.testing.assert_allclose(orig * 2, arr)

    def test_no_arg_barrier_support(self):
        # @dppy.kernel("void(float32[::1])")
        @dppy.kernel
        def twice(A):
            i = dppy.get_global_id(0)
            d = A[i]
            # no argument defaults to global mem fence
            dppy.barrier()
            A[i] = d * 2

        N = 256
        arr = np.random.random(N).astype(np.float32)
        orig = arr.copy()

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            twice[N, dppy.DEFAULT_LOCAL_SIZE](arr)

        # The computation is correct?
        np.testing.assert_allclose(orig * 2, arr)

    def test_local_memory(self):
        blocksize = 10

        # @dppy.kernel("void(float32[::1])")
        @dppy.kernel
        def reverse_array(A):
            lm = dppy.local.static_alloc(shape=10, dtype=float32)
            i = dppy.get_global_id(0)

            # preload
            lm[i] = A[i]
            # barrier local or global will both work as we only have one work group
            dppy.barrier(dppy.CLK_LOCAL_MEM_FENCE)  # local mem fence
            # write
            A[i] += lm[blocksize - 1 - i]

        arr = np.arange(blocksize).astype(np.float32)
        orig = arr.copy()

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            reverse_array[blocksize, dppy.DEFAULT_LOCAL_SIZE](arr)

        expected = orig[::-1] + orig
        np.testing.assert_allclose(expected, arr)


if __name__ == "__main__":
    unittest.main()
