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
# limitations under the License.import numpy as np

import dpctl
import numpy as np

import numba_dppy as dppy


@dppy.kernel(debug=True)
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    l1 = a[i] + 2.5
    l2 = b[i] * 0.3
    c[i] = l1 + l2


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
c = np.ones_like(a)

device = dpctl.SyclDevice("opencl:gpu")
with dpctl.device_context(device):
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)

print("Done...")
