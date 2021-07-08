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
from numba import vectorize
import dpctl
import numba_dppy as dppy


@vectorize(nopython=True)
def ufunc_kernel(x, y):
    return x + y


def get_device():
    device = None
    try:
        device = dpctl.select_gpu_device()
    except:
        try:
            device = dpctl.select_cpu_device()
        except:
            raise RuntimeError("No device found")
    return device


def main():
    N = 10
    dtype = np.float64

    A = np.arange(N, dtype=dtype)
    B = np.arange(N, dtype=dtype) * 10

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dppy.offload_to_sycl_device(device):
        C = ufunc_kernel(A, B)

    print(C)

    print("Done...")


if __name__ == "__main__":
    main()
