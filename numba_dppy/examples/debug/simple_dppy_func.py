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


@dpex.func(debug=True)
def func_sum(a_in_func, b_in_func):
    result = a_in_func + b_in_func
    return result


@dpex.kernel(debug=True)
def kernel_sum(a_in_kernel, b_in_kernel, c_in_kernel):
    i = dpex.get_global_id(0)
    c_in_kernel[i] = func_sum(a_in_kernel[i], b_in_kernel[i])


global_size = 10
a = np.arange(global_size, dtype=np.float32)
b = np.arange(global_size, dtype=np.float32)
c = np.empty_like(a)

device = dpctl.SyclDevice("opencl:gpu")
with dpctl.device_context(device):
    kernel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

print("Done...")
