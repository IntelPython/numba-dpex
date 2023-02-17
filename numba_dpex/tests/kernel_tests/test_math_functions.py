# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings

list_of_unary_ops = ["fabs", "exp", "log", "sqrt", "sin", "cos", "tan"]


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


@pytest.mark.parametrize("filter_str", filter_strings)
def test_binary_ops(filter_str, unary_op, input_arrays):
    a, actual = input_arrays
    uop = getattr(math, unary_op)
    np_uop = getattr(np, unary_op)

    @dpex.kernel
    def f(a, b):
        i = dpex.get_global_id(0)
        b[i] = uop(a[i])

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        f[a.size, dpex.DEFAULT_LOCAL_SIZE](a, actual)

    expected = np_uop(a)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
