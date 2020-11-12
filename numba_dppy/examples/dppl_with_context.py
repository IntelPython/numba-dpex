import numpy as np
from numba import njit, prange
import numba_dppy, numba_dppy as dppl
import dpctl

@njit
def add_two_arrays(b, c):
    a = np.empty_like(b)
    for i in prange(len(b)):
        a[i] = b[i] + c[i]

    return a


def main():
    N = 10
    b = np.ones(N)
    c = np.ones(N)

    if dpctl.has_gpu_queues():
        with dpctl.device_context("opencl:gpu"):
            gpu_result = add_two_arrays(b, c)
        print('GPU device found. Result on GPU:', gpu_result)
    elif dpctl.has_cpu_queues():
        with dpctl.device_context("opencl:cpu"):
            cpu_result = add_two_arrays(b, c)
        print('CPU device found. Result on CPU:', cpu_result)
    else:
        print("No device found")


if __name__ == '__main__':
    main()
