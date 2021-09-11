################################################################################
#                                 Numba-DPPY
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import dpctl
import numpy as np
import pytest
from numba import njit

import numba_dppy as dppy
from numba_dppy.tests._helper import assert_auto_offloading, skip_test

list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_binary_ops = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
]


@pytest.fixture(params=list_of_binary_ops)
def binary_op(request):
    return request.param


list_of_unary_ops = [
    "bitwise_not",
    "invert",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    np.int32,
    np.int64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


def test_binary_ops(filter_str, binary_op, input_arrays):
    if skip_test(filter_str):
        pytest.skip()

    a, b = input_arrays
    binop = getattr(np, binary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    @njit
    def f(a, b):
        return binop(a, b)

    device = dpctl.SyclDevice(filter_str)
    with dppy.offload_to_sycl_device(device), assert_auto_offloading():
        actual = f(a, b)

    expected = binop(a, b)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def test_unary_ops(filter_str, unary_op, input_arrays):
    if skip_test(filter_str):
        pytest.skip()

    a = input_arrays[0]
    uop = getattr(np, unary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    @njit
    def f(a):
        return uop(a)

    device = dpctl.SyclDevice(filter_str)
    with dppy.offload_to_sycl_device(device), assert_auto_offloading():
        actual = f(a)

    expected = uop(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
