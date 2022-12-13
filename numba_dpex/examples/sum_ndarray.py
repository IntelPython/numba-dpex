#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
from _helper import has_cpu, has_gpu

import numba_dpex as dpex


@dpex.kernel(
    access_types={
        "read_only": ["a", "b"],
        "write_only": ["c"],
        "read_write": [],
    }
)
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
local_size = 32
N = global_size * local_size

a = np.arange(N, dtype=np.float32)
b = np.arange(N, dtype=np.float32)
c = np.empty_like(a)


def main():

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        print("before A: ", a)
        print("before B: ", b)
        data_parallel_sum[global_size, local_size](a, b, c)
        print("after  C: ", c)

    print("Done...")


if __name__ == "__main__":
    main()
