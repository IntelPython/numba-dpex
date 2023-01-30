# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex.core.kernel_interface.utils import Range


@dpex.kernel(debug=True)
def data_parallel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
    i = dpex.get_global_id(0)  # numba-kernel-breakpoint
    l1 = a_in_kernel[i]  # second-line
    l2 = b_in_kernel[i]  # third-line
    c_in_kernel[i] = l1 + l2  # fourth-line


def driver(a, b, c, global_size):
    print("before : ", a)
    print("before : ", b)
    print("before : ", c)
    data_parallel_sum[Range(global_size)](a, b, c)
    print("after : ", c)


def main():
    global_size = 10
    N = global_size

    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty_like(a)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        driver(a, b, c, global_size)

    print("Done...")


if __name__ == "__main__":
    main()
