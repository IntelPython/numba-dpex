#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import float32, float64, int32, int64, njit, vectorize

from numba_dpex.tests._helper import assert_auto_offloading, filter_strings

list_of_shape = [
    (100, 100),
    (100, (10, 10)),
    (100, (2, 5, 10)),
]


@pytest.fixture(params=list_of_shape)
def shape(request):
    return request.param


@pytest.mark.xfail
@pytest.mark.parametrize("filter_str", filter_strings)
def test_njit(filter_str):
    @vectorize(nopython=True)
    def axy(a, x, y):
        return a * x + y

    def f(a0, a1):
        return np.cos(axy(a0, np.sin(a1) - 1.0, 1.0))

    A = np.random.random(10)
    B = np.random.random(10)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), assert_auto_offloading():
        f_njit = njit(f)
        expected = f_njit(A, B)
        actual = f(A, B)

        max_abs_err = expected.sum() - actual.sum()
        assert max_abs_err < 1e-5


list_of_dtype = [
    (np.int32, int32),
    (np.float32, float32),
    (np.int64, int64),
    (np.float64, float64),
]


@pytest.fixture(params=list_of_dtype)
def dtypes(request):
    return request.param


list_of_input_type = ["array", "scalar"]


@pytest.fixture(params=list_of_input_type)
def input_type(request):
    return request.param


@pytest.mark.xfail
@pytest.mark.parametrize("filter_str", filter_strings)
def test_vectorize(filter_str, shape, dtypes, input_type):
    def vector_add(a, b):
        return a + b

    dtype, sig_dtype = dtypes
    sig = [sig_dtype(sig_dtype, sig_dtype)]
    size, shape = shape

    if input_type == "array":
        A = np.arange(size, dtype=dtype).reshape(shape)
        B = np.arange(size, dtype=dtype).reshape(shape)
    elif input_type == "scalar":
        A = dtype(1.2)
        B = dtype(2.3)

    with dpctl.device_context(filter_str):
        f = vectorize(sig, target="dpex")(vector_add)
        expected = f(A, B)
        actual = vector_add(A, B)

        max_abs_err = np.sum(expected) - np.sum(actual)
        assert max_abs_err < 1e-5
