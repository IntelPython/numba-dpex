# Copyright 2021 Intel Corporation
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

from numba import njit, gdb
import numpy as np
import dpctl


@njit
def f1(a, b):
    c = a + b
    return c


N = 5
print("N", N)

a = np.ones((N, N, N, N, N), dtype=np.float32)
b = np.ones((N, N, N, N, N), dtype=np.float32)

print("a:", a, hex(a.ctypes.data))
print("b:", b, hex(b.ctypes.data))

with dpctl.device_context("opencl:gpu:0"):
    c = f1(a, b)

print("BIG RESULT c:", c, hex(c.ctypes.data))
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                for m in range(N):
                    if c[i, j, k, l, m] != 2.0:
                        print("First index not equal to 2.0 was", i, j, k, l, m)
                        break
