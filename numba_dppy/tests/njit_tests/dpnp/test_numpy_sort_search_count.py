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
from numba_dppy.context_manager import offload_to_sycl_device
import numpy as np
from numba import njit
import pytest
from .dpnp_skip_test import dpnp_skip_test as skip_test
from ._helper import wrapper_function
from numba_dppy.tests._helper import dpnp_debug


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
    N = 10
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


list_of_shape = [
    (10),
    (5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_ops = [
    "sort",
    "argmax",
    "argmin",
    "argsort",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return wrapper_function("a", f"np.{request.param}(a)", globals()), request.param


def test_unary_ops(filter_str, unary_op, input_arrays, get_shape, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = input_arrays[0]
    op, name = unary_op
    if name != "argsort" and name != "sort":
        a = np.reshape(a, get_shape)
    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with offload_to_sycl_device(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


@pytest.mark.parametrize("kth", [0, 1], ids=["0", "1"])
@pytest.mark.parametrize(
    "array",
    [
        [3, 4, 2, 1],
        [[1, 0], [3, 0]],
        [[3, 2], [1, 6]],
        [[4, 2, 3], [3, 4, 1]],
        [[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]],
        [[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]],
    ],
    ids=[
        "[3, 4, 2, 1]",
        "[[1, 0], [3, 0]]",
        "[[3, 2], [1, 6]]",
        "[[4, 2, 3], [3, 4, 1]]",
        "[[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]]",
        "[[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]",
    ],
)
def test_partition(array, kth, filter_str):
    if skip_test(filter_str):
        pytest.skip()

    a = np.array(array)

    def fn(a, kth):
        return np.partition(a, kth)

    f = njit(fn)
    device = dpctl.SyclDevice(filter_str)
    with offload_to_sycl_device(device), dpnp_debug():
        actual = f(a, kth)

    expected = fn(a, kth)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
