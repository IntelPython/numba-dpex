# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import sys

import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as dpex

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


def test_kernel_arg_types(input_arrays):
    usm_type = "device"

    a, b, c = input_arrays
    expected = a * c

    queue = dpctl.SyclQueue(dpctl.select_default_device())

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    db = dpt.usm_ndarray(
        b.shape,
        dtype=b.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

    kernel = dpex.kernel(mul_kernel)
    kernel[dpex.NdRange(dpex.Range(global_size), dpex.Range(local_size))](
        da, db, c
    )

    result = np.zeros_like(b)
    db.usm_data.copy_to_host(result.reshape((-1)).view("|u1"))

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=0)


def check_bool_kernel(A, test):
    if test:
        A[0] = 111
    else:
        A[0] = 222


def test_bool_type():
    usm_type = "device"
    a = np.array([2], np.int64)

    queue = dpctl.SyclQueue(dpctl.select_default_device())

    da = dpt.usm_ndarray(
        a.shape,
        dtype=a.dtype,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

    kernel = dpex.kernel(check_bool_kernel)

    kernel[dpex.Range(a.size)](da, True)
    result = np.zeros_like(a)
    da.usm_data.copy_to_host(result.reshape((-1)).view("|u1"))
    assert result[0] == 111

    kernel[dpex.Range(a.size)](da, False)
    result = np.zeros_like(a)
    da.usm_data.copy_to_host(result.reshape((-1)).view("|u1"))
    assert result[0] == 222
