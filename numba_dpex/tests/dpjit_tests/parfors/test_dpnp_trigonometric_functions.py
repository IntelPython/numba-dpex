# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import filter_strings, is_gen12

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
    dpnp.float32,
    dpnp.float64,
]


@pytest.fixture(params=list_of_trig_ops)
def dtype(request):
    return request.param


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048

    a = dpnp.array(dpnp.random.random(N), request.param)
    b = dpnp.array(dpnp.random.random(N), request.param)
    return a, b


@pytest.mark.parametrize("filter_str", filter_strings)
def test_trigonometric_fn(filter_str, trig_op, input_arrays):
    # FIXME: Why does archcosh fail on Gen12 discrete graphics card?
    if trig_op == "arccosh" and is_gen12(filter_str):
        pytest.skip()

    a, b = input_arrays
    trig_fn = getattr(dpnp, trig_op)
    actual = dpnp.empty(shape=a.shape, dtype=a.dtype)
    expected = dpnp.empty(shape=a.shape, dtype=a.dtype)

    if trig_op == "arctan2":

        @dpjit
        def f(a, b):
            return trig_fn(a, b)

        actual = f(a, b)
        expected = trig_fn(a, b)
    else:

        @dpjit
        def f(a):
            return trig_fn(a)

        actual = f(a)
        expected = trig_fn(a)

    np.testing.assert_allclose(
        dpt.asnumpy(actual._array_obj),
        dpt.asnumpy(expected._array_obj),
        rtol=1e-5,
        atol=0,
    )
