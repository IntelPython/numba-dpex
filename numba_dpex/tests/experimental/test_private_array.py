# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
import pytest

import numba_dpex.experimental as dpex_exp
from numba_dpex.kernel_api import Item, PrivateArray, Range
from numba_dpex.kernel_api import call_kernel as kapi_call_kernel


def private_array_kernel(item: Item, a):
    i = item.get_linear_id()
    p = PrivateArray(10, a.dtype)

    for j in range(10):
        p[j] = j * j

    a[i] = 0
    for j in range(10):
        a[i] += p[j]


def private_2d_array_kernel(item: Item, a):
    i = item.get_linear_id()
    p = PrivateArray(shape=(5, 2), dtype=a.dtype)

    for j in range(10):
        p[j % 5, j // 5] = j * j

    a[i] = 0
    for j in range(10):
        a[i] += p[j % 5, j // 5]


@pytest.mark.parametrize(
    "kernel", [private_array_kernel, private_2d_array_kernel]
)
@pytest.mark.parametrize(
    "call_kernel, decorator",
    [(dpex_exp.call_kernel, dpex_exp.kernel), (kapi_call_kernel, lambda a: a)],
)
def test_private_array(call_kernel, decorator, kernel):
    kernel = decorator(kernel)

    a = dpnp.empty(10, dtype=dpnp.float32)
    call_kernel(kernel, Range(a.size), a)

    # sum of squares from 1 to n: n*(n+1)*(2*n+1)/6
    want = np.full(a.size, (9) * (9 + 1) * (2 * 9 + 1) / 6, dtype=np.float32)

    assert np.array_equal(want, a.asnumpy())
