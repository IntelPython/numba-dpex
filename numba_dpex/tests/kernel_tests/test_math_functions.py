# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpctl
import dpnp
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings

list_of_unary_ops = [
    "fabs",
    "exp",
    "log",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "ceil",
    "floor",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
]

N = 2048


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    c = np.zeros(N, dtype=np.int64)
    return a, b, c


@pytest.mark.parametrize("filter_str", filter_strings)
def test_binary_ops(filter_str, unary_op, input_arrays):
    a, actual, actual_value_types = input_arrays

    if unary_op == "ceil" or unary_op == "floor":
        a = a * 10.0

    uop = getattr(math, unary_op)
    np_uop = getattr(np, unary_op)

    @dpex.kernel
    def f(a, b, c):
        i = dpex.get_global_id(0)
        k = uop(a[i])
        if type(k) is int or type(k) is np.int32 or type(k) is np.int64:
            c[i] = 3
        b[i] = k

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        f[a.size, dpex.DEFAULT_LOCAL_SIZE](a, actual, actual_value_types)

    expected = np_uop(a)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)

    if unary_op == "ceil" or unary_op == "floor":
        expected_value_types = (np.ones(N) * 3).astype(np.int64)
        np.testing.assert_equal(actual_value_types, expected_value_types)
