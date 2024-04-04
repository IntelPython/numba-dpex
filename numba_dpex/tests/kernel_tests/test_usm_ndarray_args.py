# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import dpnp
import numpy
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import get_all_dtypes


@dpex.kernel
def sum_2d(item, a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = item.get_id(0)
    j = item.get_id(1)
    c[i, j] = a[i, j] + b[i, j]


@dpex.kernel
def sum_2d_slice(item, a, b, c):
    """
    Vector addition using the ``kernel`` decorator.
    """
    i = item.get_id(0)
    j = item.get_id(1)
    ai, bi, ci = a[i], b[i], c[i]
    ci[j] = ai[j] + bi[j]


@pytest.mark.parametrize(
    "usm_type",
    [
        "shared",
        "device",
        "host",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    ),
)
@pytest.mark.parametrize(
    "kernel",
    [
        sum_2d,
        sum_2d_slice,
    ],
)
@pytest.mark.parametrize(
    "np",
    [
        dpt,
        dpnp,
    ],
)
def test_consuming_usm_ndarray(
    kernel,
    dtype,
    usm_type,
    np,
):
    N = 1000
    global_size = N * N

    a = np.arange(global_size, dtype=dtype, usm_type=usm_type)
    a = np.reshape(a, (N, N))

    b = np.arange(global_size, dtype=dtype, usm_type=usm_type)
    b = np.reshape(b, (N, N))

    c = np.empty_like(a)

    dpex.call_kernel(kernel, dpex.Range(N, N), a, b, c)

    na, nb, nc = np.asnumpy(a), np.asnumpy(b), np.asnumpy(c)

    assert numpy.array_equal(nc, na + nb)
