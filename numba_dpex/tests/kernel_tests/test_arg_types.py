# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings

global_size = 1054
local_size = 1
N = global_size * local_size


def mul_kernel(a, b, c):
    i = dpex.get_global_id(0)
    b[i] = a[i] * c


list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    a = np.array(np.random.random(N), request.param)
    b = np.empty_like(a, request.param)
    c = np.array([2], request.param)
    return a, b, c[0]


@pytest.mark.parametrize("filter_str", filter_strings)
def test_kernel_arg_types(filter_str, input_arrays):
    kernel = dpex.kernel(mul_kernel)
    a, actual, c = input_arrays
    expected = a * c
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[global_size, local_size](a, actual, c)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)


def check_bool_kernel(A, test):
    if test:
        A[0] = 111
    else:
        A[0] = 222


@pytest.mark.parametrize("filter_str", filter_strings)
def test_bool_type(filter_str):
    kernel = dpex.kernel(check_bool_kernel)
    a = np.array([2], np.int64)

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        kernel[a.size, dpex.DEFAULT_LOCAL_SIZE](a, True)
        assert a[0] == 111
        kernel[a.size, dpex.DEFAULT_LOCAL_SIZE](a, False)
        assert a[0] == 222
