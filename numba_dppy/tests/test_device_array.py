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
import numba_dppy as dppy
import pytest
import dpctl
import dpctl.memory as dpctl_mem
from numba_dppy.tests.skip_tests import skip_test

list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_array(request):
    # The size of input and out arrays to be used
    N = 100
    a = np.array(np.random.random(N), request.param)
    return a


list_of_shape = [
    (100,),
    (50, 2),
    (10, 5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


def test_array_creation(filter_str, input_array, get_shape):
    a = np.reshape(input_array, get_shape)

    with dpctl.device_context(filter_str):
        da = dppy.DPPYDeviceArray(a.shape, a.strides, a.dtype)
        assert da.shape == a.shape
        assert da.strides == a.strides
        assert da.dtype == a.dtype
        assert da.size == a.size
        assert da.itemsize == a.dtype.itemsize


@pytest.mark.xfail(strict=True)
def test_array_creation_with_buf(filter_str, input_array, get_shape):
    a = np.reshape(input_array, get_shape)

    with dpctl.device_context(filter_str):
        usm_buf = dpctl_mem.MemoryUSMShared(a.size * a.dtype.itemsize)
        da = dppy.DPPYDeviceArray(shape, strides, dtype, usm_memory=usm_buf)


def test_array_values(filter_str, input_array, get_shape):
    a = np.reshape(input_array, get_shape)

    with dpctl.device_context(filter_str):
        da = dppy.DPPYDeviceArray(a.shape, a.strides, a.dtype)
        da.copy_to_device(a)
        b = da.copy_to_host()
        assert np.array_equal(a, b)


def test_array_as_arg(filter_str, input_array):
    @dppy.kernel
    def data_parallel_sum(a, b, c):
        i = dppy.get_global_id(0)
        c[i] = a[i] + b[i]

    a = input_array
    b = np.ones_like(a)

    with dpctl.device_context(filter_str):
        da = dppy.DPPYDeviceArray(a.shape, a.strides, a.dtype)
        da.copy_to_device(a)

        db = dppy.DPPYDeviceArray(b.shape, b.strides, b.dtype)
        db.copy_to_device(b)

        dc = dppy.DPPYDeviceArray(a.shape, a.strides, a.dtype)

        data_parallel_sum[a.size, dppy.DEFAULT_LOCAL_SIZE](da, db, dc)

        c = dc.copy_to_host()

        assert np.allclose(c, a + b)

        # We can mix device array and ndarray
        data_parallel_sum[c.size, dppy.DEFAULT_LOCAL_SIZE](c, db, dc)
        d = dc.copy_to_host()
        assert np.allclose(d, c + b)


def test_array_api(filter_str, input_array, get_shape):
    a = np.reshape(input_array, get_shape)

    with dpctl.device_context(filter_str):
        da = dppy.DPPYDeviceArray(a.shape, a.strides, a.dtype)

        da.copy_to_device(a)
        b = da.copy_to_host()
        c = np.empty_like(a)
        da.copy_to_host(c)
        assert np.array_equal(b, c)
