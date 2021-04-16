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
    func_str = "def fn(a):\n    return a." + request.param + "()"
    ldict = {}
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    return fn, request.param


def test_unary_ops(filter_str, unary_op, input_arrays, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_arrays[0]
    op, name = unary_op
    if name != "argsort" and name != "copy":
        a = np.reshape(a, get_shape)
    if name == "cumprod" and (
        filter_str == "opencl:cpu:0" or a.dtype == np.int32 or is_gen12(filter_str)
    ):
        pytest.skip()
    if name == "cumsum" and (
        filter_str == "opencl:cpu:0" or a.dtype == np.int32 or is_gen12(filter_str)
    ):
        pytest.skip()
    if name == "mean" and is_gen12(filter_str):
        pytest.skip()
    if name == "argmax" and is_gen12(filter_str):
        pytest.skip()

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    with dpctl.device_context(filter_str), dpnp_debug():
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
    func_str = "def fn(a, ind):\n    return a.take(ind)"
    ldict = {}
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    return fn


def test_take(filter_str, input_arrays, indices, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_arrays[0]
    fn = get_take_fn()

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(fn)
    with dpctl.device_context(filter_str), dpnp_debug():
        actual = f(a, indices)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = fn(a, indices)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
