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
from numba_dppy.tests._helper import filter_strings

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


@pytest.mark.parametrize("filter_str", filter_strings)
def test_dpnp_create_array_in_context(filter_str, dtype):
    import dpnp

    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in filter_str
    ):
        pytest.skip("Bug in DPNP. See: IntelPython/dpnp#723")

    with dpctl.device_context(filter_str):
        a = dpnp.arange(1024, dtype=dtype)  # noqa


@pytest.mark.parametrize("filter_str", filter_strings)
def test_consuming_array_from_dpnp(filter_str, dtype):
    import dpnp

    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in filter_str
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

    with dpctl.device_context(filter_str):
        a = dppy.asarray(dpnp.arange(global_size, dtype=dtype))
        b = dppy.asarray(dpnp.arange(global_size, dtype=dtype))
        c = dppy.asarray(dpnp.ones_like(a))

        data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
