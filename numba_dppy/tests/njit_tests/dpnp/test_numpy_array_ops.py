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
from numba_dppy.tests._helper import (
    dpnp_debug,
    filter_strings,
    is_gen12,
    skip_no_dpnp,
    skip_windows,
)

from ._helper import wrapper_function

pytestmark = skip_no_dpnp

list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 10
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


list_of_shape = [
    (10),
    (5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_ops = [
    "sum",
    "prod",
    "max",
    "min",
    "mean",
    "argmax",
    "argmin",
    "argsort",
    "copy",
    "cumsum",
    "cumprod",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return (
        wrapper_function("a", f"a.{request.param}()", globals()),
        request.param,
    )


@pytest.mark.parametrize("filter_str", filter_strings)
def test_unary_ops(filter_str, unary_op, input_arrays, get_shape, capfd):
    a = input_arrays[0]
    op, name = unary_op
    if name != "argsort" and name != "copy":
        a = np.reshape(a, get_shape)
    if name == "cumprod" and (
        filter_str == "opencl:cpu:0"
        or a.dtype == np.int32
        or is_gen12(filter_str)
    ):
        pytest.skip()
    if name == "cumsum" and (
        filter_str == "opencl:cpu:0"
        or a.dtype == np.int32
        or is_gen12(filter_str)
    ):
        pytest.skip()
    if name == "mean" and is_gen12(filter_str):
        pytest.skip()
    if name == "argmax" and is_gen12(filter_str):
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
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


list_of_indices = [
    np.array([0, 2, 5]),
    np.array([0, 5]),
]


@pytest.fixture(params=list_of_indices)
def indices(request):
    return request.param


def get_take_fn():
    return wrapper_function("a, ind", "a.take(ind)", globals())


@skip_windows
@pytest.mark.parametrize("filter_str", filter_strings)
def test_take(filter_str, input_arrays, indices, capfd):
    a = input_arrays[0]
    fn = get_take_fn()

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(fn)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, indices)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = fn(a, indices)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
