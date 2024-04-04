# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpnp
import numpy
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes

list_of_unary_ops = ["fabs", "exp", "log", "sqrt", "sin", "cos", "tan"]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = get_all_dtypes(
    no_bool=True, no_int=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048
    a = dpnp.arange(N, dtype=request.param)
    b = dpnp.arange(N, dtype=request.param)
    return a, b


def test_binary_ops(unary_op, input_arrays):
    a, b = input_arrays
    uop = getattr(math, unary_op)
    dpnp_uop = getattr(dpnp, unary_op)

    @dpex.kernel
    def f(item, a, b):
        i = item.get_id(0)
        b[i] = uop(a[i])

    dpex.call_kernel(f, dpex.Range(a.size), a, b)

    expected = dpnp_uop(a)

    np_expected = dpnp.asnumpy(expected)
    np_actual = dpnp.asnumpy(b)

    assert numpy.allclose(np_expected, np_actual)
