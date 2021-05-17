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


def test_ufunc():
    N = 10
    dtype = np.float64

    A = np.arange(N, dtype=dtype)
    B = np.arange(N, dtype=dtype) * 10

    device = get_device()
    with dpctl.device_context(device):
        C = ufunc_kernel(A, B)

    print(C)


if __name__ == "__main__":
    test_ufunc()
