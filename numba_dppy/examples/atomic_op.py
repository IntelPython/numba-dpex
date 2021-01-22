import numpy as np

import numba
import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


def main():
    @dppy.kernel
    def atomic_add(a):
        dppy.atomic.add(a, 0, 1)

    global_size = 100
    a = np.array([0])

    with dpctl.device_context("opencl:gpu") as gpu_queue:
        atomic_add[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
        # Expected 100, because global_size = 100
        print(a)


if __name__ == "__main__":
    main()
