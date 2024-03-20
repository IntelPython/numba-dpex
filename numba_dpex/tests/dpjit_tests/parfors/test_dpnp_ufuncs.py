# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.core.typing.dpnpdecl import (
    _bit_twiddling_functions,
    _comparison_functions,
    _floating_functions,
    _logic_functions,
    _math_operations,
    _trigonometric_functions,
    _unsupported,
)
from numba_dpex.tests._helper import (
    get_float_dtypes,
    get_int_dtypes,
    is_gen12,
    num_required_arguments,
)

int_ops = {
    "gcd",
    "lcm",
    "mod",
    "divmod",
} | set(_bit_twiddling_functions)


float_ops = {
    "fmod",
    "reciprocal",
}

all_ufuncs = set(
    _math_operations
    + _trigonometric_functions
    + _bit_twiddling_functions
    + _comparison_functions
    + _floating_functions
    + _logic_functions
) - {
    "frexp"  # there is no dpnp.frexp
}

unary_ops = {
    op for op in all_ufuncs if num_required_arguments(getattr(dpnp, op)) == 1
} | {"frexp"}

binary_ops = {
    op for op in all_ufuncs if num_required_arguments(getattr(dpnp, op)) == 2
}

float_ops |= {
    "copysign",
    "divide",
    "hypot",
    "power",
    "true_divide",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "ceil",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "erf",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "log",
    "log10",
    "log1p",
    "log2",
    "rad2deg",
    "radians",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
}

int_dtypes = get_int_dtypes()
float_dtypes = get_float_dtypes(no_float16=True)

unary_op_dtypes = set(product(unary_ops - int_ops, float_dtypes)) | set(
    product(unary_ops - float_ops, int_dtypes)
)
binary_op_dtypes = set(product(binary_ops - int_ops, float_dtypes)) | set(
    product(binary_ops - float_ops, int_dtypes)
)

N = 1024


@pytest.mark.parametrize(
    "binary_op, dtype",
    sorted(list(binary_op_dtypes), key=lambda a: (a[0], str(a[1]))),
)
def test_binary_ops(binary_op, dtype):
    # TODO: fails for float32 because it uses cast to float64 internally?
    if binary_op == "arctan2" and dpnp.float64 not in float_dtypes:
        pytest.xfail("arctan2 requires float64 support")

    a = dpnp.array(dpnp.random.random(N), dtype)
    b = dpnp.array(dpnp.random.random(N), dtype)

    binop = getattr(dpnp, binary_op)
    actual = dpnp.empty(shape=a.shape, dtype=a.dtype)
    expected = dpnp.empty(shape=a.shape, dtype=a.dtype)

    if binary_op in _unsupported:
        pytest.xfail(reason="not supported")

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


@pytest.mark.parametrize(
    "unary_op, dtype",
    sorted(list(unary_op_dtypes), key=lambda a: (a[0], str(a[1]))),
)
def test_unary_ops(unary_op, dtype):
    if unary_op in _unsupported:
        pytest.xfail(reason="not supported")

    xfail_ops = ["sign", "log", "log2", "log10", "expm1", "arccosh"]
    if unary_op in xfail_ops and is_gen12():
        pytest.xfail(f"{unary_op} does not work on gen12")

    a = dpnp.array(dpnp.random.random(N), dtype)

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
