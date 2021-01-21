from __future__ import print_function, division, absolute_import

import numpy as np
import math
import time

import numba_dppy, numba_dppy as dppy
import dpctl


@dppy.kernel
def reduction_kernel(A, R, stride):
    i = dppy.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def reduce(A, R):
    """Work only for size = power of two"""
    total = len(A)

    context = get_context()
    with dpctl.device_context(context):
        while total > 1:
            # call kernel
            global_size = total // 2
            reduction_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](
                A, R, global_size
            )
            total = total // 2

    return R[0]


def test_sum_reduction():
    # This test will only work for size = power of two
    N = 2048
    assert N % 2 == 0

    A = np.array(np.random.random(N), dtype=np.float32)
    A_copy = A.copy()
    # at max we will require half the size of A to store sum
    R = np.array(np.random.random(math.ceil(N / 2)), dtype=np.float32)

    actual = reduce(A, R)
    expected = A_copy.sum()
    max_abs_err = expected - actual

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert max_abs_err < 1e-2


if __name__ == "__main__":
    test_sum_reduction()
