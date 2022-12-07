# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex


def main():
    """
    The example demonstrates the use of numba_dpex's ``atomic_add`` intrinsic
    function on a SYCL GPU device. The ``dpctl.select_gpu_device`` is
    equivalent to ``sycl::gpu_selector`` and returns a sycl::device of type GPU.

    If we want to generate native floating point atomics for supported
    SYCL devices we need to set two environment variables:
    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1

    To run this example:
    NUMBA_DPEX_ACTIVATE_ATOMICS_FP_NATIVE=1 python atomic_op.py

    Without these two environment variables numba_dpex will use other
    implementation for floating point atomics.
    """

    @dpex.kernel
    def atomic_add(a):
        dpex.atomic.add(a, 0, 1)

    global_size = 100
    a = np.array([0], dtype=np.float32)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        atomic_add[global_size, dpex.DEFAULT_LOCAL_SIZE](a)

    # Expected 100, because global_size = 100
    print(a)

    print("Done...")


if __name__ == "__main__":
    main()
