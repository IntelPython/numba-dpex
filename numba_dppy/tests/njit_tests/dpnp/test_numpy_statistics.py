################################################################################
#                                 Numba-DPPY
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import dpctl
import numpy as np
import pytest
from numba import njit

import numba_dppy as dppy
from numba_dppy.tests._helper import (
    dpnp_debug,
    filter_strings_with_skips_for_opencl,
)

from ._helper import wrapper_function
from .dpnp_skip_test import dpnp_skip_test as skip_test

list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 100
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


list_of_shape = [
    (100),
    (50, 2),
    (10, 5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_ops = [
    "max",
    "amax",
    "min",
    "amin",
    "median",
    "mean",
    "cov",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return (
        wrapper_function("a", f"np.{request.param}(a)", globals()),
        request.param,
    )


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_unary_ops(filter_str, unary_op, input_arrays, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_arrays[0]
    op, name = unary_op
    if name != "cov":
        a = np.reshape(a, get_shape)

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
