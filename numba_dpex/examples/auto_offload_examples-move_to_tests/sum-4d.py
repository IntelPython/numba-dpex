# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
from numba import njit


@njit
def f1(a, b):
    c = a + b
    return c


N = 10
print("N", N)

a = np.ones((N, N, N, N), dtype=np.float32)
b = np.ones((N, N, N, N), dtype=np.float32)

print("a:", a, hex(a.ctypes.data))
print("b:", b, hex(b.ctypes.data))


def main():
    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        c = f1(a, b)

    print("c:", c, hex(c.ctypes.data))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):  # noqa
                    if c[i, j, k, l] != 2.0:
                        print("First index not equal to 2.0 was", i, j, k, l)
                        break

    print("Done...")


if __name__ == "__main__":
    main()
