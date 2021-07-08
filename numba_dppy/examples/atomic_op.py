# Copyright 2020, 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import numba_dppy as dppy
import dpctl


def main():
    """
    The example demonstrates the use of numba_dppy's ``atomic_add`` intrinsic
    function on a SYCL GPU device. The ``dpctl.select_gpu_device`` is
    equivalent to ``sycl::gpu_selector`` and returns a sycl::device of type GPU.

    If we want to generate native floating point atomics for spported
    SYCL devices we need to set two environment variables:
    NUMBA_DPPY_ACTIVATE_ATOMCIS_FP_NATIVE=1
    NUMBA_DPPY_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv

    To run this example:
    NUMBA_DPPY_ACTIVATE_ATOMCIS_FP_NATIVE=1 NUMBA_DPPY_LLVM_SPIRV_ROOT=/path/to/dpcpp/provided/llvm_spirv python atomic_op.py

    Without these two environment variables Numba_dppy will use other
    implementation for floating point atomics.
    """

    @dppy.kernel
    def atomic_add(a):
        dppy.atomic.add(a, 0, 1)

    global_size = 100
    a = np.array([0], dtype=np.float32)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dppy.offload_to_sycl_device(device):
        atomic_add[global_size, dppy.DEFAULT_LOCAL_SIZE](a)

    # Expected 100, because global_size = 100
    print(a)

    print("Done...")


if __name__ == "__main__":
    main()
