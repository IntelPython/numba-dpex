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
import numba
import argparse

import numba_dppy as dppy


def func(param_a, param_b):
    param_c = param_a + 10  # Set breakpoint
    param_d = param_b * 0.5
    result = param_c + param_d
    return result


dppy_func = dppy.func(debug=True)(func)
numba_func = numba.njit(debug=True)(func)


@dppy.kernel(debug=True)
def dppy_kernel(a_in_kernel, b_in_kernel, c_in_kernel):
    i = dppy.get_global_id(0)
    c_in_kernel[i] = dppy_func(a_in_kernel[i], b_in_kernel[i])


@numba.njit(debug=True)
def numba_func_driver(a, b, c):
    for i in range(len(c)):
        c[i] = numba_func(a[i], b[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dppy',
                        required=False,
                        action='store_true',
                        help="Start the dppy version of functions",
    )

    args = parser.parse_args()

    global_size = 10
    N = global_size

    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty_like(a)

    if args.dppy:
        device = dpctl.select_default_device()
        with dppy.offload_to_sycl_device(device):
            dppy_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
    else:
        numba_func_driver(a, b, c)

    print("Done...")


if __name__ == "__main__":
    main()
