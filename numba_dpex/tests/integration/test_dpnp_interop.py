# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import numba_dpex as dpex

dpnp = pytest.importorskip("dpnp", reason="DPNP is not installed")


list_of_dtype = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtype)
def dtype(request):
    return request.param


def test_consuming_array_from_dpnp(dtype):
    @dpex.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 1021

    a = dpnp.arange(global_size, dtype=dtype)
    b = dpnp.arange(global_size, dtype=dtype)
    c = dpnp.ones_like(a)

    data_parallel_sum[global_size](a, b, c)
