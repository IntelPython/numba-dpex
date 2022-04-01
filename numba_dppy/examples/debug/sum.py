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

import dpctl
import numpy as np

import numba_dppy as dpex


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
    data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)
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
