# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numba as nb
import numpy
import pytest

import numba_dpex as dpex

N = 10


@dpex.dpjit
def vecadd_prange1(a, b):
    s = 0
    t = 0
    for i in nb.prange(a.shape[0]):
        s += a[i] + b[i]
    for i in nb.prange(a.shape[0]):
        t += a[i] - b[i]
    return s - t


@dpex.dpjit
def vecadd_prange2(a, b):
    t = 0
    for i in nb.prange(a.shape[0]):
        t += a[i] * b[i]
    return t


@dpex.dpjit
def vecmul_prange(a, b):
    t = 1
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


list_of_dtypes = [
    dpnp.int32,
    dpnp.int64,
    dpnp.float64,
]


@pytest.fixture(params=list_of_dtypes)
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


def test_dpjit_array_arg_float32_types(input_arrays):
    """Tests passing float32 type dpnp arrays to a dpjit
    prange function.Local variable has to be casted to float32.

    Args:
        input_arrays (dpnp.ndarray): Array arguments to be passed to a kernel.
    """
    s = 9900
    a = dpnp.arange(100, dtype=dpnp.float32)
    b = dpnp.arange(100, dtype=dpnp.float32)

    c = vecadd_prange_float(a, b)

    assert s == c
