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
import pytest
from numba import njit
from numba.tests.support import captured_stdout

import numba_dpex as dpex
from numba_dpex import config

from ._helper import (
    assert_auto_offloading,
    filter_strings,
    skip_no_opencl_cpu,
    skip_no_opencl_gpu,
)


def scenario(filter_str, context):
    @njit
    def func(a, b):
        return a + b

    a = np.ones((64), dtype=np.int32)
    b = np.ones((64), dtype=np.int32)

    device = dpctl.SyclDevice(filter_str)
    with context(device):
        func(a, b)


@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize(
    "context",
    [
        dpex.offload_to_sycl_device,
        dpctl.device_context,
    ],
)
def test_dpctl_device_context_affects_numba_pipeline(filter_str, context):
    with assert_auto_offloading():
        scenario(filter_str, context)


class TestWithDeviceContext:
    @skip_no_opencl_gpu
    def test_with_device_context_gpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        config.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_gpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_gpu_message:
            device = dpctl.SyclDevice("opencl:gpu")
            with dpctl.device_context(device):
                func(got_gpu)

        config.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_gpu)
        assert "Parfor offloaded to opencl:gpu" in got_gpu_message.getvalue()

    @skip_no_opencl_cpu
    def test_with_device_context_cpu(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        config.DEBUG = 1
        expected = np.ones((64), dtype=np.float64)
        got_cpu = np.ones((64), dtype=np.float64)

        with captured_stdout() as got_cpu_message:
            device = dpctl.SyclDevice("opencl:cpu")
            with dpctl.device_context(device):
                func(got_cpu)

        config.DEBUG = 0
        func(expected)

        np.testing.assert_array_equal(expected, got_cpu)
        assert "Parfor offloaded to opencl:cpu" in got_cpu_message.getvalue()
