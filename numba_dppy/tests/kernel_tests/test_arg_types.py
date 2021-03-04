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
from numba_dppy.context import device_context
from numba_dppy.tests.skip_tests import skip_test

global_size = 1054
local_size = 1
N = global_size * local_size


def mul_kernel(a, b, c):
    i = dppy.get_global_id(0)
    b[i] = a[i] * c


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
def input_arrays(request):
    a = np.array(np.random.random(N), request.param)
    b = np.empty_like(a, request.param)
    c = np.array([2], request.param)
    return a, b, c[0]


def test_kernel_arg_types(filter_str, input_arrays):
    if skip_test(filter_str):
        pytest.skip()

    kernel = dppy.kernel(mul_kernel)
    a, actual, c = input_arrays
    expected = a * c
    with device_context(filter_str):
        kernel[global_size, local_size](a, actual, c)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def check_bool_kernel(A, test):
    if test:
        A[0] = 111
    else:
        A[0] = 222


def test_bool_type(filter_str):
    if skip_test(filter_str):
        pytest.skip()

    kernel = dppy.kernel(check_bool_kernel)
    a = np.array([2], np.int64)

    with device_context(filter_str):
        kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](a, True)
        assert a[0] == 111
        kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](a, False)
        assert a[0] == 222
