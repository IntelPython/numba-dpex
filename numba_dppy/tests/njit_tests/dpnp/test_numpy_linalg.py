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

from ._helper import args_string, wrapper_function
from .dpnp_skip_test import dpnp_skip_test as skip_test

filter_strings = [
    "level_zero:gpu:0",
    "opencl:gpu:0",
    "opencl:cpu:0",
]


# From https://github.com/IntelPython/dpnp/blob/0.4.0/tests/test_linalg.py#L8
def vvsort(val, vec):
    size = val.size
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            if np.abs(val[imax]) < np.abs(val[j]):
                imax = j

        temp = val[i]
        val[i] = val[imax]
        val[imax] = temp

        if not (vec is None):
            for k in range(size):
                temp = vec[k, i]
                vec[k, i] = vec[k, imax]
                vec[k, imax] = temp


def get_fn(name, nargs):
    args = args_string(nargs)
    return wrapper_function(args, f"np.{name}({args})", globals())


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def eig_input(request):
    # The size of input and out arrays to be used
    N = 10
    a = np.arange(N * N, dtype=request.param).reshape((N, N))
    symm_a = (
        np.tril(a)
        + np.tril(a, -1).T
        + +np.diag(np.full((N,), N * N, dtype=request.param))
    )
    return symm_a


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_eig(filter_str, eig_input, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = eig_input
    fn = get_fn("linalg.eig", 1)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual_val, actual_vec = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected_val, expected_vec = fn(a)

        # sort val/vec by abs value
        vvsort(actual_val, actual_vec)
        vvsort(expected_val, expected_vec)

        # NP change sign of vectors
        for i in range(expected_vec.shape[1]):
            if expected_vec[0, i] * actual_vec[0, i] < 0:
                expected_vec[:, i] = -expected_vec[:, i]

        assert np.allclose(actual_val, expected_val)
        assert np.allclose(actual_vec, expected_vec)


@pytest.fixture(params=list_of_dtypes)
def dtype(request):
    return request.param


list_of_dim = [
    (10, 1, 10, 1),
    (10, 1, 10, 2),
    (2, 10, 10, 1),
    (10, 2, 2, 10),
]


@pytest.fixture(params=list_of_dim)
def dot_input(request):
    # The size of input and out arrays to be used
    a1, a2, b1, b2 = request.param
    a = np.array(np.random.random(a1 * a2))
    b = np.array(np.random.random(b1 * b2))
    if a2 != 1:
        a = a.reshape(a1, a2)
    if b2 != 1:
        b = b.reshape(b1, b2)

    return a, b


list_of_dot_name = ["dot", "vdot"]


@pytest.fixture(params=list_of_dot_name)
def dot_name(request):
    return request.param


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_dot(filter_str, dot_name, dot_input, dtype, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a, b = dot_input

    if dot_name == "vdot":
        if a.size != b.size:
            pytest.skip("vdot only supports same sized arrays")

    a = a.astype(dtype)
    b = b.astype(dtype)
    fn = get_fn(dot_name, 2)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, b)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(a, b)
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_matmul(filter_str, dtype, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.array(np.random.random(10 * 2), dtype=dtype).reshape(10, 2)
    b = np.array(np.random.random(2 * 10), dtype=dtype).reshape(2, 10)
    fn = get_fn("matmul", 2)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, b)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(a, b)
        assert np.allclose(actual, expected)


@pytest.mark.skip(reason="dpnp does not support it yet")
def test_cholesky(filter_str, dtype, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.array([[1, -2], [2, 5]], dtype=dtype)
    fn = get_fn("linalg.cholesky", 1)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(a)
        assert np.allclose(actual, expected)


list_of_det_input = [
    [[0, 0], [0, 0]],
    [[1, 2], [1, 2]],
    [[1, 2], [3, 4]],
    [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
    [
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
        [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
    ],
]


@pytest.fixture(params=list_of_det_input)
def det_input(request):
    return request.param


@pytest.mark.parametrize("filter_str", filter_strings)
def test_det(filter_str, det_input, dtype, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.array(det_input, dtype=dtype)
    fn = get_fn("linalg.det", 1)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(a)
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_multi_dot(filter_str, capfd):
    if skip_test(filter_str):
        pytest.skip()

    def fn(A, B, C, D):
        c = np.linalg.multi_dot([A, B, C, D])
        return c

    A = np.random.random((10000, 100))
    B = np.random.random((100, 1000))
    C = np.random.random((1000, 5))
    D = np.random.random((5, 333))
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(A, B, C, D)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(A, B, C, D)
        assert np.allclose(actual, expected)


list_of_power = [2, 3, 0]


@pytest.fixture(params=list_of_power)
def power(request):
    return request.param


list_of_matrix_power_input = [
    [[0, 0], [0, 0]],
    [[1, 2], [1, 2]],
    [[1, 2], [3, 4]],
]


@pytest.fixture(params=list_of_matrix_power_input)
def matrix_power_input(request):
    return request.param


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_matrix_power(filter_str, matrix_power_input, power, dtype, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = np.array(matrix_power_input, dtype=dtype)
    fn = get_fn("linalg.matrix_power", 2)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a, power)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(a, power)
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize(
    "matrix_rank_input",
    [
        pytest.param(
            np.eye(4),
            marks=pytest.mark.xfail(reason="dpnp does not support it yet"),
        ),
        np.ones((4,)),
        pytest.param(
            np.ones((4, 4)),
            marks=pytest.mark.xfail(reason="dpnp does not support it yet"),
        ),
        np.zeros((4,)),
    ],
)
def test_matrix_rank(filter_str, matrix_rank_input, capfd):
    if skip_test(filter_str):
        pytest.skip()

    fn = get_fn("linalg.matrix_rank", 1)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(matrix_rank_input)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected = fn(matrix_rank_input)
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_eigvals(filter_str, eig_input, capfd):
    if skip_test(filter_str):
        pytest.skip()

    a = eig_input
    fn = get_fn("linalg.eigvals", 1)
    f = njit(fn)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual_val = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        expected_val = fn(a)

        # sort val/vec by abs value
        vvsort(actual_val, None)
        vvsort(expected_val, None)

        assert np.allclose(actual_val, expected_val)
