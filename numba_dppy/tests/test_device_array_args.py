#! /usr/bin/env python
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

import dpctl
import numpy as np

import numba_dppy as dpex
from numba_dppy.tests._helper import skip_no_opencl_cpu, skip_no_opencl_gpu


@dpex.kernel
def data_parallel_sum(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


global_size = 64
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)
b = np.array(np.random.random(N), dtype=np.float32)
d = a + b


@skip_no_opencl_cpu
class TestDPPYDeviceArrayArgsGPU:
    def test_device_array_args_cpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:cpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

            assert np.all(c == d)


@skip_no_opencl_gpu
class TestDPPYDeviceArrayArgsCPU:
    def test_device_array_args_gpu(self):
        c = np.ones_like(a)

        with dpctl.device_context("opencl:gpu"):
            data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)

        assert np.all(c == d)
