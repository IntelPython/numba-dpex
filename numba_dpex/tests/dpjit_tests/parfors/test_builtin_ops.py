# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import dpctl
import dpnp
import numpy
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import (
    get_float_dtypes,
    get_int_dtypes,
    has_level_zero,
    has_opencl_gpu,
    is_windows,
    num_required_arguments,
)


def parfor_add(a, b):
    "Same as a + b."
    return a + b


def parfor_and_(a, b):
    "Same as a & b."
    return a & b


def parfor_floordiv(a, b):
    "Same as a // b."
    return a // b


def parfor_inv(a):
    "Same as ~a."
    return ~a


def parfor_lshift(a, b):
    "Same as a << b."
    return a << b


def parfor_mod(a, b):
    "Same as a % b."
    return a % b


def parfor_mul(a, b):
    "Same as a * b."
    return a * b


def parfor_matmul(a, b):
    "Same as a @ b."
    return a @ b


def parfor_neg(a):
    "Same as -a."
    return -a


def parfor_or_(a, b):
    "Same as a | b."
    return a | b


def parfor_pos(a):
    "Same as +a."
    return +a


def parfor_pow(a, b):
    "Same as a ** b."
    return a**b


def parfor_rshift(a, b):
    "Same as a >> b."
    return a >> b


def parfor_sub(a, b):
    "Same as a - b."
    return a - b


def parfor_truediv(a, b):
    "Same as a / b."
    return a / b


def parfor_xor(a, b):
    "Same as a ^ b."
    return a ^ b


# Inplace operations
def parfor_iadd(a, b):
    "Same as a += b."
    a += b
    return a


def parfor_iand(a, b):
    "Same as a &= b."
    a &= b
    return a


def parfor_ifloordiv(a, b):
    "Same as a //= b."
    a //= b
    return a


def parfor_ilshift(a, b):
    "Same as a <<= b."
    a <<= b
    return a


def parfor_imod(a, b):
    "Same as a %= b."
    a %= b
    return a


def parfor_imul(a, b):
    "Same as a *= b."
    a *= b
    return a


def parfor_imatmul(a, b):
    "Same as a @= b."
    a @= b
    return a


def parfor_ior(a, b):
    "Same as a |= b."
    a |= b
    return a


def parfor_ipow(a, b):
    "Same as a **= b."
    a **= b
    return a


def parfor_irshift(a, b):
    "Same as a >>= b."
    a >>= b
    return a


def parfor_isub(a, b):
    "Same as a -= b."
    a -= b
    return a


def parfor_itruediv(a, b):
    "Same as a /= b."
    a /= b
    return a


def parfor_ixor(a, b):
    "Same as a ^= b."
    a ^= b
    return a


# Comparison operations


def parfor_lt(a, b):
    "Same as a < b."
    return a < b


def parfor_le(a, b):
    "Same as a <= b."
    return a <= b


def parfor_eq(a, b):
    "Same as a == b."
    return a == b


def parfor_ne(a, b):
    "Same as a != b."
    return a != b


def parfor_ge(a, b):
    "Same as a >= b."
    return a >= b


def parfor_gt(a, b):
    "Same as a > b."
    return a > b


# logical


def parfor_not_(a):
    "Same as not a."
    return not a


shapes = [100, (25, 4)]
int_dtypes = set(get_int_dtypes())
float_dtypes = set(get_float_dtypes())
usm_types = ["device", "host", "shared"]

funcs = {
    parfor_add,
    parfor_and_,
    parfor_floordiv,
    parfor_inv,
    parfor_lshift,
    parfor_mod,
    parfor_mul,
    parfor_matmul,
    parfor_neg,
    parfor_or_,
    parfor_pos,
    parfor_pow,
    parfor_rshift,
    parfor_sub,
    parfor_truediv,
    parfor_xor,
}

inplace_funcs = {
    parfor_iand,
    parfor_iadd,
    parfor_iand,
    parfor_ifloordiv,
    parfor_ilshift,
    parfor_imod,
    parfor_imul,
    parfor_imatmul,
    parfor_ior,
    parfor_ipow,
    parfor_irshift,
    parfor_isub,
    parfor_itruediv,
    parfor_ixor,
}

comparison_funcs = {
    parfor_lt,
    parfor_le,
    parfor_eq,
    parfor_ne,
    parfor_ge,
    parfor_gt,
}

all_funcs = funcs | inplace_funcs | comparison_funcs

not_float_funcs = {
    parfor_inv,
    parfor_and_,
    parfor_floordiv,
    parfor_lshift,
    parfor_mod,
    parfor_or_,
    parfor_rshift,
    parfor_xor,
    parfor_iand,
    parfor_ifloordiv,
    parfor_ilshift,
    parfor_imod,
    parfor_ior,
    parfor_irshift,
    parfor_ixor,
}

not_int_funcs = {
    parfor_itruediv,
}

not_bool_funcs = {
    parfor_neg,
    parfor_pos,
    parfor_sub,
    parfor_lshift,
    parfor_rshift,
    parfor_mod,
    parfor_pow,
    parfor_floordiv,
    parfor_isub,
    parfor_ilshift,
    parfor_irshift,
    parfor_imod,
    parfor_ipow,
    parfor_ifloordiv,
} | not_int_funcs

unsupported_funcs = {
    parfor_matmul,
    parfor_imatmul,
    parfor_not_,  # not supported by numpy
}


if has_opencl_gpu() or is_windows() or has_level_zero():
    unsupported_funcs |= {parfor_ipow}


func_dtypes = (
    set(product(all_funcs - not_float_funcs, float_dtypes))
    | set(product(all_funcs - not_int_funcs, int_dtypes))
    | set(product((all_funcs | {parfor_not_}) - not_bool_funcs, {dpnp.bool}))
)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize(
    "func, dtype",
    sorted(list(func_dtypes), key=lambda a: (str(a[0]), str(a[1]))),
)
def test_built_in_operators(shape, dtype, usm_type, func):
    if func in unsupported_funcs:
        pytest.xfail(reason="not supported")

    if dpnp.float64 not in float_dtypes and func in [
        parfor_truediv,
        parfor_pow,
    ]:
        pytest.xfail(
            f"{func} does not work on without double precision support"
        )

    queue = dpctl.SyclQueue()
    a = dpnp.zeros(
        shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
    )
    b = dpnp.ones(shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
    try:
        op = dpjit(func)
        if num_required_arguments(func) == 1:
            c = op(a)
        else:
            c = op(a, b)
        del op
    except Exception:
        pytest.fail("Failed to compile.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    if func not in {parfor_truediv} | comparison_funcs:
        assert c.dtype == dtype
    assert c.usm_type == usm_type

    assert c.sycl_device.filter_string == queue.sycl_device.filter_string

    if num_required_arguments(func) == 1:
        expected = dpnp.asnumpy(func(a))
    else:
        expected = dpnp.asnumpy(func(a, b))
    nc = dpnp.asnumpy(c)

    numpy.allclose(nc, expected)
