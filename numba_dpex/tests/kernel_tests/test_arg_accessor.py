# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
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
