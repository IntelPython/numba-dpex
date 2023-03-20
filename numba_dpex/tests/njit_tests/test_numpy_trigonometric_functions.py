# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dpex.tests._helper import (
    assert_auto_offloading,
    filter_strings,
    is_gen12,
)

list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_trig_ops = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "degrees",
    "radians",
]


@pytest.fixture(params=list_of_trig_ops)
def trig_op(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_trig_ops)
def dtype(request):
    return request.param


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048
    # Note: These inputs do not work for all of the functions and
    # can result in warnings. E.g. arccosh needs the range of values
    # to be greater than 0, while arccos needs them to be [-1,1].
    # These warnings are relatively benign as NumPy will return "nan"
    # for such cases.
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


@pytest.mark.parametrize("filter_str", filter_strings)
def test_trigonometric_fn(filter_str, trig_op, input_arrays):
    # FIXME: Why does archcosh fail on Gen12 discrete graphics card?
    if trig_op == "arccosh" and is_gen12(filter_str):
        pytest.skip()

    a, b = input_arrays
    trig_fn = getattr(np, trig_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    if trig_op == "arctan2":

        @njit
        def f(a, b):
            return trig_fn(a, b)

        device = dpctl.SyclDevice(filter_str)
        with dpctl.device_context(device), assert_auto_offloading():
            actual = f(a, b)
        expected = trig_fn(a, b)
    else:

        @njit
        def f(a):
            return trig_fn(a)

        device = dpctl.SyclDevice(filter_str)
        with dpctl.device_context(device), assert_auto_offloading():
            actual = f(a)
        expected = trig_fn(a)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
