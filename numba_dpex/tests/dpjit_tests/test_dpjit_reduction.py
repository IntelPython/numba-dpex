# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numba as nb
import numpy
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes

N = 10


@dpex.dpjit
def vecadd_prange1(a, b):
    s = a.dtype.type(0)
    t = a.dtype.type(0)
    for i in nb.prange(a.shape[0]):
        s += a[i] + b[i]
    for i in nb.prange(a.shape[0]):
        t += a[i] - b[i]
    return s - t


@dpex.dpjit
def vecadd_prange2(a, b):
    t = a.dtype.type(0)
    for i in nb.prange(a.shape[0]):
        t += a[i] * b[i]
    return t


@dpex.dpjit
def vecmul_prange(a, b):
    t = a.dtype.type(1)
    for i in nb.prange(a.shape[0]):
        t *= a[i] + b[i]
    return t


@dpex.dpjit
def vecadd_prange_float(a, b):
    s = numpy.float32(0)
    t = numpy.float32(0)
    for i in nb.prange(a.shape[0]):
        s += a[i] + b[i]
    for i in nb.prange(a.shape[0]):
        t += a[i] - b[i]
    return s - t


@pytest.fixture(
    params=get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    )
)
def input_arrays(request):
    a = dpnp.arange(N, dtype=request.param)
    b = dpnp.ones(N, dtype=request.param)

    return a, b


def test_dpjit_array_arg_types_add1(input_arrays):
    """Tests passing float and int type dpnp arrays to a dpjit
    prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    s = 20
    a, b = input_arrays
    c = vecadd_prange1(a, b)

    assert s == c


def test_dpjit_array_arg_types_add2(input_arrays):
    """Tests passing float and int type dpnp arrays to a dpjit
    prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    t = 45
    a, b = input_arrays
    d = vecadd_prange2(a, b)

    assert t == d


def test_dpjit_array_arg_types_mul(input_arrays):
    """Tests passing float and int type dpnp arrays to a dpjit
    prange function.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    s = 3628800

    a, b = input_arrays

    c = vecmul_prange(a, b)

    assert s == c
