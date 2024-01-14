# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit

list_of_binary_ops = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "right_shift",
]


@pytest.fixture(params=list_of_binary_ops)
def binary_op(request):
    return request.param


list_of_unary_ops = [
    "bitwise_not",
    "invert",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    dpnp.int32,
    dpnp.int64,
]


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
    a = input_arrays[0]
    uop = getattr(dpnp, unary_op)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

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
