# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from itertools import permutations

import dpctl
import dpnp
import numpy as np
import pytest
from numpy.testing import assert_equal

import numba_dpex as dpex
from numba_dpex import NdRange, Range
from numba_dpex.experimental.kernel_iface.kernel_sim.kernel import (
    kernel as kernel_sim,
)
from numba_dpex.tests._helper import filter_strings


def call_kernel(global_size, local_size, A, B, C, func):
    func[global_size, local_size](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size


def sum_kernel(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    c = np.zeros_like(a)
    return a, b, c


list_of_kernel_opt = [
    {"read_only": ["a", "b"], "write_only": ["c"], "read_write": []},
    {},
]


@pytest.fixture(params=list_of_kernel_opt)
def kernel(request):
    return dpex.kernel(access_types=request.param)(sum_kernel)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_kernel_arg_accessor(filter_str, input_arrays, kernel):
    a, b, actual = input_arrays
    expected = a + b
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        call_kernel(global_size, local_size, a, b, actual, kernel)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


@pytest.mark.parametrize("device", filter_strings)
@pytest.mark.parametrize(
    "shape",
    [
        Range(1),
        Range(7),
        Range(1, 1),
        Range(7, 13),
        Range(1, 1, 1),
        Range(7, 13, 23),
    ],
)
def test_get_global_id(device, shape):
    def func1(c):
        i = dpex.get_global_id(0)
        c[i] = i

    def func2(c):
        i = dpex.get_global_id(0)
        j = dpex.get_global_id(1)
        c[i, j] = i + j * 100

    def func3(c):
        i = dpex.get_global_id(0)
        j = dpex.get_global_id(1)
        k = dpex.get_global_id(2)
        c[i, j, k] = i + j * 100 + k * 10000

    func = [func1, func2, func3][len(shape) - 1]

    sim_func = kernel_sim(func)
    gpu_func = dpex.kernel(func)

    dtype = np.int32

    sim_res = np.zeros(shape, dtype)
    sim_func[shape](sim_res)

    gpu_res = dpnp.zeros(shape, dtype=dtype, device=device)
    gpu_func[shape](gpu_res)

    assert_equal(gpu_res, sim_res)


def _get_ndrange_permutations():
    dims = [6, 12, 24]

    result = []
    globals_sizes = (
        list(permutations(dims, 1))
        + list(permutations(dims, 2))
        + list(permutations(dims, 3))
    )
    for gz in globals_sizes:
        local_sizes = list(permutations([1, 3], len(gz)))
        for lz in local_sizes:
            if any([lv > gv for gv, lv in zip(gz, lz)]):
                continue

            result.append(NdRange(gz, lz))

    return result


_ndranges = _get_ndrange_permutations()


@pytest.mark.parametrize("device", filter_strings)
@pytest.mark.parametrize("shape", _ndranges, ids=list(map(str, _ndranges)))
def test_get_local_id(device, shape):
    def func1(c):
        i = dpex.get_global_id(0)
        li = dpex.get_local_id(0)
        c[i] = li

    def func2(c):
        i = dpex.get_global_id(0)
        j = dpex.get_global_id(1)
        li = dpex.get_local_id(0)
        lj = dpex.get_local_id(1)
        c[i, j] = li + lj * 100

    def func3(c):
        i = dpex.get_global_id(0)
        j = dpex.get_global_id(1)
        k = dpex.get_global_id(2)
        li = dpex.get_local_id(0)
        lj = dpex.get_local_id(1)
        lk = dpex.get_local_id(2)
        c[i, j, k] = li + lj * 100 + lk * 10000

    global_size = shape.global_range
    func = [func1, func2, func3][len(global_size) - 1]

    sim_func = kernel_sim(func)
    gpu_func = dpex.kernel(func)

    dtype = np.int32

    sim_res = np.zeros(global_size, dtype)
    sim_func[shape](sim_res)

    gpu_res = dpnp.zeros(global_size, dtype=dtype, device=device)
    gpu_func[shape](gpu_res)

    assert_equal(gpu_res, sim_res)
