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

from numba_dppy.tests._helper import dpnp_debug, filter_strings, is_gen12

from ._helper import wrapper_function
from .dpnp_skip_test import skip_no_dpnp

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


def get_func(name):
    return wrapper_function("a", f"np.{name}(a)", globals())


@pytest.fixture(params=list_of_unary_ops + list_of_nan_ops)
def unary_op(request):
    fn = get_func(request.param)
    return fn, request.param


@pytest.fixture(params=list_of_nan_ops)
def unary_nan_op(request):
    fn = get_func(request.param)
    return fn, request.param


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_unary_ops(filter_str, unary_op, input_array, get_shape, capfd):
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
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    max_abs_err = np.sum(actual - expected)
    assert max_abs_err < 1e-4


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_unary_nan_ops(
    filter_str, unary_nan_op, input_nan_array, get_shape, capfd
):
    a = input_nan_array
    a = np.reshape(a, get_shape)
    op, name = unary_nan_op
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    if name == "nansum" and is_gen12(filter_str):
        pytest.skip()

    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    max_abs_err = np.sum(actual - expected)
    assert max_abs_err < 1e-4
