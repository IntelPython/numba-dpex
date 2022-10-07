# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings

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


@pytest.mark.parametrize("filter_str", filter_strings)
def test_dpnp_create_array_in_context(filter_str, dtype):
    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in filter_str
    ):
        pytest.skip("Bug in DPNP. See: IntelPython/dpnp#723")

    with dpctl.device_context(filter_str):
        a = dpnp.arange(1024, dtype=dtype)  # noqa


@pytest.mark.parametrize("filter_str", filter_strings)
def test_consuming_array_from_dpnp(filter_str, dtype):
    if (
        "opencl" not in dpctl.get_current_queue().sycl_device.filter_string
        and "opencl" in filter_str
    ):
        pytest.skip("Bug in DPNP. See: IntelPython/dpnp#723")

    @dpex.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 1021

    with dpctl.device_context(filter_str):
        a = dpex.asarray(dpnp.arange(global_size, dtype=dtype))
        b = dpex.asarray(dpnp.arange(global_size, dtype=dtype))
        c = dpex.asarray(dpnp.ones_like(a))

        data_parallel_sum[global_size, dpex.DEFAULT_LOCAL_SIZE](a, b, c)
