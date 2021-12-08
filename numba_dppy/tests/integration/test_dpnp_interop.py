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

import numba_dppy as dppy
from numba_dppy.tests._helper import ensure_dpnp, skip_test

dpnp = pytest.importorskip("dpnp", reason="DPNP is not installed")


list_of_dtype = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtype)
def dtype(request):
    return request.param


list_of_usm_type = [
    "shared",
    "device",
    "host",
]


@pytest.fixture(params=list_of_usm_type)
def usm_type(request):
    return request.param


def test_dpnp_create_array_in_context(offload_device, dtype):
    if skip_test(offload_device):
        pytest.skip("No device for " + offload_device)

    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in offload_device
    ):
        pytest.skip("Bug in DPNP. See: IntelPython/dpnp#723")

    with dpctl.device_context(offload_device):
        a = dpnp.arange(1024, dtype=dtype)  # noqa


def test_consuming_array_from_dpnp(offload_device, dtype):
    if skip_test(offload_device):
        pytest.skip("No device for " + offload_device)

    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in offload_device
    ):
        pytest.skip("Bug in DPNP. See: IntelPython/dpnp#723")

    @dppy.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dppy.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 1021

    with dpctl.device_context(offload_device):
        a = dppy.asarray(dpnp.arange(global_size, dtype=dtype))
        b = dppy.asarray(dpnp.arange(global_size, dtype=dtype))
        c = dppy.asarray(dpnp.ones_like(a))

        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
