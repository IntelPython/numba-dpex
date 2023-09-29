# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import numpy
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes


def parfor_add(a, b):
    return a + b


def parfor_sub(a, b):
    return a - b


def parfor_mult(a, b):
    return a * b


def parfor_divide(a, b):
    return a / b


def parfor_modulus(a, b):
    return a % b


def parfor_exponent(a, b):
    return a**b


shapes = [100, (25, 4)]
dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)
usm_types = ["device"]
funcs = [
    parfor_add,
    parfor_sub,
    parfor_mult,
    parfor_divide,
    parfor_modulus,
    parfor_exponent,
]

# TODO: fails for integer because it is being cast to float64 internally?
if dpnp.float64 not in dtypes:
    funcs.remove(parfor_divide)
    funcs.remove(parfor_exponent)


def parfor_floor(a, b):
    return a // b


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("func", funcs)
def test_built_in_operators1(shape, dtype, usm_type, func):
    queue = dpctl.SyclQueue()
    a = dpnp.zeros(
        shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
    )
    b = dpnp.ones(shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
    try:
        op = dpjit(func)
        c = op(a, b)
        del op
    except Exception:
        pytest.fail("Failed to compile.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    if func != parfor_divide:
        assert c.dtype == dtype
    assert c.usm_type == usm_type

    assert c.sycl_device.filter_string == queue.sycl_device.filter_string

    expected = dpnp.asnumpy(func(a, b))
    nc = dpnp.asnumpy(c)

    numpy.allclose(nc, expected)


usm_types = ["host", "shared"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("func", funcs)
def test_built_in_operators2(shape, dtype, usm_type, func):
    queue = dpctl.SyclQueue()
    a = dpnp.zeros(
        shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue
    )
    b = dpnp.ones(shape=shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
    try:
        op = dpjit(func)
        c = op(a, b)
        del op
    except Exception:
        pytest.fail("Failed to compile.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    if func != parfor_divide:
        assert c.dtype == dtype
    assert c.usm_type == usm_type

    assert c.sycl_device.filter_string == queue.sycl_device.filter_string

    expected = dpnp.asnumpy(func(a, b))
    nc = dpnp.asnumpy(c)

    numpy.allclose(nc, expected)
