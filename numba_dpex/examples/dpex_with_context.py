# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The numba_dpex extension adds an automatic offload optimizer to
numba. The optimizer automatically detects data-parallel code
regions in a numba.jit function and then offloads the data-parallel
regions to a SYCL device. The optimizer is triggered when a numba.jit
function is invoked inside ``dpctl.device_context`` scope.

This example demonstrates the usage of numba_dpex's automatic offload
functionality. Note that numba_dpex should be installed in your
environment for the example to work.
"""

import dpctl
import numpy as np
from numba import njit, prange


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

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        result = add_two_arrays(b, c)

    print("Result :", result)

    print("Done...")


if __name__ == "__main__":
    main()
