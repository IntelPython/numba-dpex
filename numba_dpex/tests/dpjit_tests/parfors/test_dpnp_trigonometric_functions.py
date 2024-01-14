# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes, is_gen12

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

list_of_dtypes = get_all_dtypes(
    no_bool=True, no_int=True, no_float16=True, no_none=True, no_complex=True
)

# TODO: fails for float32 because it uses cast to float64 internally?
if dpnp.float64 not in list_of_dtypes:
    list_of_trig_ops.remove("arctan2")


@pytest.fixture(params=list_of_trig_ops)
def trig_op(request):
    return request.param


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048

    a = dpnp.array(dpnp.random.random(N), request.param)
    b = dpnp.array(dpnp.random.random(N), request.param)
    return a, b


def test_trigonometric_fn(trig_op, input_arrays):
    filter_str = dpctl.SyclDevice().filter_string
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
