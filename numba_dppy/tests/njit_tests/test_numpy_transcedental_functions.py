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

from numba_dppy.context import device_context
import numpy as np
from numba import njit
import pytest
from numba_dppy.tests.skip_tests import skip_test, is_gen12

list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_binary_ops = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "power",
    "remainder",
    "mod",
    "fmod",
    "hypot",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
]


@pytest.fixture(params=list_of_binary_ops)
def binary_op(request):
    return request.param


list_of_unary_ops = [
    "negative",
    "abs",
    "absolute",
    "fabs",
    "sign",
    "conj",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "reciprocal",
    "conjugate",
    "floor",
    "ceil",
    "trunc",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
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

    with device_context(filter_str):
        actual = f(a, b)

    expected = binop(a, b)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def test_unary_ops(filter_str, unary_op, input_arrays):
    if skip_test(filter_str):
        pytest.skip()

    # FIXME: Why does sign fail on Gen12 discrete graphics card?
    skip_ops = ["sign", "log", "log2", "log10", "expm1"]
    if unary_op in skip_ops and is_gen12(filter_str):
        pytest.skip()

    a = input_arrays[0]
    uop = getattr(np, unary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    @njit
    def f(a):
        return uop(a)

    with device_context(filter_str):
        actual = f(a)

    expected = uop(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
