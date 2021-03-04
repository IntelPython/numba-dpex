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
from numba_dppy.tests.skip_tests import skip_test

import numba_dppy as dppy
import pytest

from numba_dppy.context import device_context


global_size = 100
N = global_size


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
def return_dtype(request):
    return request.param


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = np.array([0], request.param)
    return a, request.param


list_of_op = [
    ("add", N),
    ("sub", -N),
]


@pytest.fixture(params=list_of_op)
def return_list_of_op(request):
    return request.param[0], request.param[1]


@pytest.fixture(params=list_of_op)
def kernel_result_pair(request):
    op = getattr(dppy.atomic, request.param[0])

    def f(a):
        op(a, 0, 1)

    return dppy.kernel(f), request.param[1]


def atomic_skip_test(device_type):
    skip = False
    if skip_test(device_type):
        skip = True

    if not skip:
        if not dppy.ocl.atomic_support_present():
            skip = True

    return skip


def test_kernel_atomic_simple(filter_str, input_arrays, kernel_result_pair):
    if atomic_skip_test(filter_str):
        pytest.skip()

    a, dtype = input_arrays
    kernel, expected = kernel_result_pair
    with device_context(filter_str):
        kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected


def get_kernel_local(op_type, dtype):
    op = getattr(dppy.atomic, op_type)

    def f(a):
        lm = dppy.local.array(1, dtype)
        lm[0] = a[0]
        dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
        op(lm, 0, 1)
        dppy.barrier(dppy.CLK_GLOBAL_MEM_FENCE)
        a[0] = lm[0]

    return dppy.kernel(f)


def test_kernel_atomic_local(filter_str, input_arrays, return_list_of_op):
    if atomic_skip_test(filter_str):
        pytest.skip()

    a, dtype = input_arrays
    op_type, expected = return_list_of_op
    kernel = get_kernel_local(op_type, dtype)
    with device_context(filter_str):
        kernel[global_size, global_size](a)
    assert a[0] == expected


list_of_dim = [
    (1,),
    (1, 1),
    (1, 1, 1),
]


@pytest.fixture(params=list_of_dim)
def return_list_of_dim(request):
    return request.param


def get_kernel_multi_dim(op_type, size):
    op = getattr(dppy.atomic, op_type)
    if size == 1:
        idx = 0
    else:
        idx = (0,)
        for i in range(size - 1):
            idx += (0,)

    def f(a):
        op(a, idx, 1)

    return dppy.kernel(f)


def test_kernel_atomic_multi_dim(
    filter_str, return_list_of_op, return_list_of_dim, return_dtype
):
    if atomic_skip_test(filter_str):
        pytest.skip()

    op_type, expected = return_list_of_op
    dim = return_list_of_dim
    kernel = get_kernel_multi_dim(op_type, len(dim))
    a = np.zeros(dim, return_dtype)
    with device_context(filter_str):
        kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](a)
    assert a[0] == expected
