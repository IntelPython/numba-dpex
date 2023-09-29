# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes, is_gen12

"""dpnp raise error on : mod, abs and remainder(float32)"""
list_of_binary_ops = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    #   "remainder",
    #   "mod",
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
    #   "abs",
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
    "erf",
]


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
    a = dpnp.array(dpnp.random.random(N), request.param)
    b = dpnp.array(dpnp.random.random(N), request.param)
    return a, b


def test_binary_ops(binary_op, input_arrays):
    a, b = input_arrays
    binop = getattr(dpnp, binary_op)
    actual = dpnp.empty(shape=a.shape, dtype=a.dtype)
    expected = dpnp.empty(shape=a.shape, dtype=a.dtype)

    @dpjit
    def f(a, b):
        return binop(a, b)

    actual = f(a, b)

    expected = binop(a, b)

    np.testing.assert_allclose(
        dpt.asnumpy(actual._array_obj),
        dpt.asnumpy(expected._array_obj),
        rtol=1e-5,
        atol=0,
    )


def test_unary_ops(unary_op, input_arrays):
    skip_ops = ["abs", "sign", "log", "log2", "log10", "expm1"]
    device = dpctl.SyclDevice()
    if unary_op in skip_ops and is_gen12(device.filter_string):
        pytest.skip()

    a = input_arrays[0]
    uop = getattr(dpnp, unary_op)
    actual = dpnp.empty(shape=a.shape, dtype=a.dtype)
    expected = dpnp.empty(shape=a.shape, dtype=a.dtype)

    @dpjit
    def f(a):
        return uop(a)

    actual = f(a)

    expected = uop(a)

    np.testing.assert_allclose(
        dpt.asnumpy(actual._array_obj),
        dpt.asnumpy(expected._array_obj),
        rtol=1e-5,
        atol=0,
    )
