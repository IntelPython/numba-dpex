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

import numpy as np
import numba_dppy, numba_dppy as dppy
import dpctl
import unittest
import math


@dppy.kernel
def dppy_fabs(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.fabs(a[i])


@dppy.kernel
def dppy_exp(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.exp(a[i])


@dppy.kernel
def dppy_log(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.log(a[i])


@dppy.kernel
def dppy_sqrt(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.sqrt(a[i])


@dppy.kernel
def dppy_sin(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.sin(a[i])


@dppy.kernel
def dppy_cos(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.cos(a[i])


@dppy.kernel
def dppy_tan(a, b):
    i = dppy.get_global_id(0)
    b[i] = math.tan(a[i])


global_size = 10
N = global_size

a = np.array(np.random.random(N), dtype=np.float32)


def driver(a, jitfunc):
    b = np.ones_like(a)
    # Device buffers
    jitfunc[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b)
    return b


def check_driver(input_arr, device_ty, jitfunc):
    out_actual = None
    if device_ty == "GPU":
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            out_actual = driver(input_arr, jitfunc)
    elif device_ty == "CPU":
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            out_actual = driver(input_arr, jitfunc)
    else:
        print("Unknown device type")
        raise SystemExit()

    return out_actual


@unittest.skipUnless(dpctl.has_cpu_queues(), "test only on CPU system")
class TestDPPYMathFunctionsCPU(unittest.TestCase):
    def test_fabs_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_cos_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_exp_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_sqrt_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_log_cpu(self):
        b_actual = check_driver(a, "CPU", dppy_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual, b_expected))


@unittest.skipUnless(dpctl.has_gpu_queues(), "test only on GPU system")
class TestDPPYMathFunctionsGPU(unittest.TestCase):
    def test_fabs_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_fabs)
        b_expected = np.fabs(a)
        self.assertTrue(np.all(b_actual == b_expected))

    def test_sin_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_sin)
        b_expected = np.sin(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_cos_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_cos)
        b_expected = np.cos(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_exp_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_exp)
        b_expected = np.exp(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_sqrt_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_sqrt)
        b_expected = np.sqrt(a)
        self.assertTrue(np.allclose(b_actual, b_expected))

    def test_log_gpu(self):
        b_actual = check_driver(a, "GPU", dppy_log)
        b_expected = np.log(a)
        self.assertTrue(np.allclose(b_actual, b_expected))


if __name__ == "__main__":
    unittest.main()
