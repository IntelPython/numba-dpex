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

import warnings

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex
from numba_dpex.tests._helper import (
    filter_strings,
    skip_no_level_zero_gpu,
    skip_no_opencl_gpu,
)
from numba_dpex.utils import (
    IndeterminateExecutionQueueError,
    IndeterminateExecutionQueueError_msg,
    cfd_ctx_mgr_wrng_msg,
    mix_datatype_err_msg,
)

global_size = 10
local_size = 1
N = global_size * local_size


@numba_dpex.kernel
def sum_kernel(a, b, c):
    i = numba_dpex.get_global_id(0)
    c[i] = a[i] + b[i]


list_of_uniform_types = [
    (np.array, np.array, np.array),
    (
        dpctl.tensor.usm_ndarray,
        dpctl.tensor.usm_ndarray,
        dpctl.tensor.usm_ndarray,
    ),
]

list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    c = np.zeros_like(a)
    return a, b, c


@pytest.mark.parametrize("offload_device", filter_strings)
def test_usm_ndarray_argtype(offload_device, input_arrays):
    usm_type = "device"

    a, b, expected = input_arrays
    got = np.ones_like(a)

    device = dpctl.SyclDevice(offload_device)
    queue = dpctl.SyclQueue(device)

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    dc = dpt.usm_ndarray(
        got.shape,
        dtype=got.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )

    sum_kernel[global_size, local_size](da, db, dc)

    dc.usm_data.copy_to_host(got.reshape((-1)).view("|u1"))

    expected = a + b

    assert np.array_equal(got, expected)


@pytest.mark.parametrize("offload_device", filter_strings)
def test_ndarray_argtype(offload_device, input_arrays):
    a, b, expected = input_arrays
    got = np.ones_like(a)

    with numba_dpex.offload_to_sycl_device(offload_device):
        sum_kernel[global_size, local_size](a, b, got)

    expected = a + b

    assert np.array_equal(got, expected)


@pytest.mark.parametrize("offload_device", filter_strings)
def test_mix_argtype(offload_device, input_arrays):
    usm_type = "device"

    a, b, expected = input_arrays
    got = np.ones_like(a)

    device = dpctl.SyclDevice(offload_device)
    queue = dpctl.SyclQueue(device)

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    dc = dpt.usm_ndarray(
        got.shape,
        dtype=got.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )

    with pytest.raises(TypeError) as error_msg:
        sum_kernel[global_size, local_size](da, b, dc)

        assert mix_datatype_err_msg in error_msg


@pytest.mark.parametrize("offload_device", filter_strings)
def test_context_manager_with_usm_ndarray(offload_device, input_arrays):
    usm_type = "device"

    a, b, expected = input_arrays
    got = np.ones_like(a)

    device = dpctl.SyclDevice(offload_device)
    queue = dpctl.SyclQueue(device)

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    dc = dpt.usm_ndarray(
        got.shape,
        dtype=got.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )

    with pytest.warns(Warning) as warning:
        with numba_dpex.offload_to_sycl_device(offload_device):
            sum_kernel[global_size, local_size](da, db, dc)
        if not warning:
            pytest.fail("Warning expected!")

    sum_kernel[global_size, local_size](da, db, dc)

    dc.usm_data.copy_to_host(got.reshape((-1)).view("|u1"))

    expected = a + b

    assert np.array_equal(got, expected)


@skip_no_level_zero_gpu
@skip_no_opencl_gpu
def test_equivalent_usm_ndarray(input_arrays):
    usm_type = "device"

    a, b, expected = input_arrays
    got = np.ones_like(a)

    device1 = dpctl.SyclDevice("level_zero:gpu")
    queue1 = dpctl.SyclQueue(device1)

    device2 = dpctl.SyclDevice("opencl:gpu")
    queue2 = dpctl.SyclQueue(device2)

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue1},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    not_equivalent_db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue2},
    )
    not_equivalent_db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    equivalent_db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue1},
    )
    equivalent_db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    dc = dpt.usm_ndarray(
        got.shape,
        dtype=got.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue1},
    )

    with pytest.raises(IndeterminateExecutionQueueError) as error_msg:
        sum_kernel[global_size, local_size](da, not_equivalent_db, dc)
        assert IndeterminateExecutionQueueError_msg in str(error_msg.value)

    sum_kernel[global_size, local_size](da, equivalent_db, dc)
    dc.usm_data.copy_to_host(got.reshape((-1)).view("|u1"))

    expected = a + b
    assert np.array_equal(got, expected)
