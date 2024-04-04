# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
import pytest

import numba_dpex as dpex
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


def private_array_kernel_fill_true(item: Item, a):
    i = item.get_linear_id()
    p = PrivateArray(10, a.dtype, fill_zeros=True)

    for j in range(10):
        p[j] = j * j

    a[i] = 0
    for j in range(10):
        a[i] += p[j]


def private_array_kernel_fill_false(item: Item, a):
    i = item.get_linear_id()
    p = PrivateArray(10, a.dtype, fill_zeros=False)

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
    "kernel",
    [
        private_array_kernel,
        private_array_kernel_fill_true,
        private_array_kernel_fill_false,
        private_2d_array_kernel,
    ],
)
@pytest.mark.parametrize(
    "call_kernel, decorator",
    [(dpex.call_kernel, dpex.kernel), (kapi_call_kernel, lambda a: a)],
)
def test_private_array(call_kernel, decorator, kernel):
    kernel = decorator(kernel)

    a = dpnp.empty(10, dtype=dpnp.float32)
    call_kernel(kernel, Range(a.size), a)

    # sum of squares from 1 to n: n*(n+1)*(2*n+1)/6
    want = np.full(a.size, (9) * (9 + 1) * (2 * 9 + 1) / 6, dtype=np.float32)

    assert np.array_equal(want, a.asnumpy())


@pytest.mark.parametrize(
    "func",
    [
        private_array_kernel,
        private_array_kernel_fill_true,
        private_array_kernel_fill_false,
        private_2d_array_kernel,
    ],
)
def test_private_array_in_device_func(func):

    _df = dpex.device_func(func)

    @dpex.kernel
    def _kernel(item: Item, a):
        _df(item, a)

    a = dpnp.empty(10, dtype=dpnp.float32)
    dpex.call_kernel(_kernel, Range(a.size), a)

    # sum of squares from 1 to n: n*(n+1)*(2*n+1)/6
    want = np.full(a.size, (9) * (9 + 1) * (2 * 9 + 1) / 6, dtype=np.float32)

    assert np.array_equal(want, a.asnumpy())
