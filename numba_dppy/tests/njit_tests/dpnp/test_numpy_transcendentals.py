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
from numba import njit
import pytest
from numba_dppy.testing import dpnp_debug
from .dpnp_skip_test import dpnp_skip_test as skip_test
from numba_dppy.tests.skip_tests import is_gen12

list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_int_dtypes = [
    np.int32,
    np.int64,
]


list_of_float_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_int_dtypes + list_of_float_dtypes)
def input_array(request):
    # The size of input and out arrays to be used
    N = 100
    a = np.array(np.random.random(N), request.param)
    return a


@pytest.fixture(params=list_of_float_dtypes)
def input_nan_array(request):
    # The size of input and out arrays to be used
    N = 100
    a = np.array(np.random.random(N), request.param)
    for i in range(5):
        a[N - 1 - i] = np.nan
    return a


list_of_shape = [
    (100),
    (50, 2),
    (10, 5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_ops = [
    "sum",
    "prod",
    "cumsum",
    "cumprod",
]

list_of_nan_ops = [
    "nansum",
    "nanprod",
]


def get_func(param):
    name = param
    func_str = "def fn(a):\n    return np." + name + "(a)"
    ldict = {}
    exec(func_str, globals(), ldict)
    return ldict["fn"]


@pytest.fixture(params=list_of_unary_ops + list_of_nan_ops)
def unary_op(request):
    fn = get_func(request.param)
    return fn, request.param


@pytest.fixture(params=list_of_nan_ops)
def unary_nan_op(request):
    fn = get_func(request.param)
    return fn, request.param


def test_unary_ops(filter_str, unary_op, input_array, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_array
    a = np.reshape(a, get_shape)
    op, name = unary_op
    if (name == "cumprod" or name == "cumsum") and (
        filter_str == "opencl:cpu:0" or is_gen12(filter_str)
    ):
        pytest.skip()
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    with dpctl.device_context(filter_str), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    max_abs_err = np.sum(actual - expected)
    assert max_abs_err < 1e-4


def test_unary_nan_ops(filter_str, unary_nan_op, input_nan_array, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_nan_array
    a = np.reshape(a, get_shape)
    op, name = unary_nan_op
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    if name == "nansum" and is_gen12(filter_str):
        pytest.skip()

    f = njit(op)
    with dpctl.device_context(filter_str), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    max_abs_err = np.sum(actual - expected)
    assert max_abs_err < 1e-4
