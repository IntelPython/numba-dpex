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

import sys
import numpy as np
import numba_dppy, numba_dppy as dppy
import dpctl
import unittest


def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


class TestCaching(unittest.TestCase):
    def test_caching_kernel(self):
        global_size = 10
        N = global_size

        a = np.array(np.random.random(N), dtype=np.float32)
        b = np.array(np.random.random(N), dtype=np.float32)
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu") as gpu_queue:
            func = dppy.kernel(data_parallel_sum)
            caching_kernel = func[global_size, dppy.DEFAULT_LOCAL_SIZE].specialize(
                a, b, c
            )

            for i in range(10):
                cached_kernel = func[global_size, dppy.DEFAULT_LOCAL_SIZE].specialize(
                    a, b, c
                )
                self.assertIs(caching_kernel, cached_kernel)


if __name__ == "__main__":
    unittest.main()
