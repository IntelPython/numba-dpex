#! /usr/bin/env python
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numba import njit, vectorize
import dpctl
import pytest

list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_shape = [
    100,
    (10, 10),
    (2, 5, 10),
]


@pytest.fixture(params=list_of_shape)
def shape(request):
    return request.param


def test_njit(filter_str):
    @vectorize(nopython=True)
    def axy(a, x, y):
        return a * x + y

    def f(a0, a1):
        return np.cos(axy(a0, np.sin(a1) - 1.0, 1.0))

    A = np.random.random(10)
    B = np.random.random(10)

    with dpctl.device_context(filter_str):
        f_njit = njit(f)
        expected = f_njit(A, B)
        actual = f(A, B)

        max_abs_err = expected.sum() - actual.sum()
        assert max_abs_err < 1e-5


def test_vectorize(filter_str, shape):
    def axy(a):
        return a + 1

    A = np.arange(100).reshape(shape)

    with dpctl.device_context(filter_str):
        f = vectorize(target="dppy")(axy)
        expected = f(A)
        actual = axy(A)

        max_abs_err = expected.sum() - actual.sum()
        assert max_abs_err < 1e-5
