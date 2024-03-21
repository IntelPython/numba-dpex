# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import numpy
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes

list_of_dtype = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=list_of_dtype)
def dtype(request):
    return request.param


list_of_usm_type = [
    "shared",
    "device",
    "host",
]


@pytest.fixture(params=list_of_usm_type)
def usm_type(request):
    return request.param


def test_consuming_usm_ndarray(dtype, usm_type):
    @dpex.kernel
    def data_parallel_sum(item, a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = item.get_id(0)
        j = item.get_id(1)
        c[i, j] = a[i, j] + b[i, j]

    N = 1000
    global_size = N * N

    a = dpt.arange(global_size, dtype=dtype, usm_type=usm_type)
    a = dpt.reshape(a, shape=(N, N))

    b = dpt.arange(global_size, dtype=dtype, usm_type=usm_type)
    b = dpt.reshape(b, shape=(N, N))

    c = dpt.empty_like(a)

    dpex.call_kernel(data_parallel_sum, dpex.Range(N, N), a, b, c)

    na = dpt.asnumpy(a)
    nb = dpt.asnumpy(b)
    nc = dpt.asnumpy(c)

    assert numpy.array_equal(nc, na + nb)
