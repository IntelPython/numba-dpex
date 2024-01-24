#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from numba import float32, float64, int32, int64, vectorize

list_of_shape = [
    (100, 100),
    (100, (10, 10)),
    (100, (2, 5, 10)),
]


@pytest.fixture(params=list_of_shape)
def shape(request):
    return request.param


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
def test_vectorize(shape, dtypes, input_type):
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

    f = vectorize(sig, target="dpex")(vector_add)
    expected = f(A, B)
    actual = vector_add(A, B)

    max_abs_err = np.sum(expected) - np.sum(actual)
    assert max_abs_err < 1e-5
