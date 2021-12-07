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
from numba_dppy.tests._helper import dpnp_debug, skip_test

from ._helper import args_string, wrapper_function

list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
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
    N = 10
    a = np.array(np.random.random(N), request.param)
    return a


list_of_shape = [
    (10),
    (5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_op = [
    "copy",
    "trace",
]

list_of_binary_op = [
    "ones_like",
    "zeros_like",
]


@pytest.fixture(params=list_of_unary_op)
def unary_op(request):
    return request.param


@pytest.fixture(params=list_of_binary_op)
def binary_op(request):
    return request.param


def get_op_fn(name, nargs):
    args = args_string(nargs)
    return wrapper_function(args, f"np.{name}({args})", globals())


def test_unary_ops(filter_str, unary_op, input_array, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_array
    if unary_op == "trace":
        a = input_array.reshape((2, 5))
    fn = get_op_fn(unary_op, 1)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(fn)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = fn(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


@pytest.fixture(params=list_of_dtypes + [None])
def dtype(request):
    return request.param


def test_binary_op(filter_str, binary_op, input_array, dtype, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.reshape(input_array, get_shape)
    fn = get_op_fn(binary_op, 2)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(fn)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, dtype)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = fn(a, dtype)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


list_of_full = [
    "full_like",
]


@pytest.fixture(params=list_of_full)
def full_name(request):
    return request.param


def test_full(filter_str, full_name, input_array, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.reshape(input_array, get_shape)
    fn = get_op_fn(full_name, 2)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(fn)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, np.array([2]))
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = fn(a, np.array([2]))
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
