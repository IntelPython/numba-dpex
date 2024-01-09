# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex
import numba_dpex.experimental as exp_dpex
from numba_dpex import NdRange, Range


@exp_dpex.kernel
def change_values_1d(x, v):
    i = numba_dpex.get_global_id(0)
    p = x[i]  # getitem
    p = v
    x[i] = p  # setitem


@exp_dpex.kernel
def change_values_2d(x, v):
    i = numba_dpex.get_global_id(0)
    j = numba_dpex.get_global_id(1)
    p = x[i, j]  # getitem
    p = v
    x[i, j] = p  # setitem


@exp_dpex.kernel
def change_values_3d(x, v):
    i = numba_dpex.get_global_id(0)
    j = numba_dpex.get_global_id(1)
    k = numba_dpex.get_global_id(2)
    p = x[i, j, k]  # getitem
    p = v
    x[i, j, k] = p  # setitem


def test_strided_dpnp_array_in_kernel():
    """
    Tests if we can correctly handle a strided 1d dpnp array
    inside dpex kernel.
    """
    N = 1024
    out = dpnp.arange(0, N * 2, dtype=dpnp.int64)
    b = out[::2]

    r = Range(N)
    v = -3
    exp_dpex.call_kernel(change_values_1d, r, b, v)

    assert (dpnp.asnumpy(b) == v).all()


def test_multievel_strided_dpnp_array_in_kernel():
    """
    Tests if we can correctly handle a multilevel strided 1d dpnp array
    inside dpex kernel.
    """
    N = 128
    out = dpnp.arange(0, N * 2, dtype=dpnp.int64)

    K = 7
    b = out
    for _ in range(K):
        b = b[::2]

    r = Range(int(N / (2 * (K - 1))))
    v = -3
    exp_dpex.call_kernel(change_values_1d, r, b, v)

    assert (dpnp.asnumpy(b) == v).all()


def test_multilevel_2d_strided_dpnp_array_in_kernel():
    """
    Tests if we can correctly handle a multilevel strided 2d dpnp array
    inside dpex kernel.
    """
    N = 128
    out, _ = dpnp.mgrid[0 : N * 2, 0 : N * 2]  # noqa: E203

    K = 7
    b = out
    for _ in range(K):
        b = b[::2, ::2]

    r = Range(int(N / (2 * (K - 1))), int(N / (2 * (K - 1))))
    v = -3
    exp_dpex.call_kernel(change_values_2d, r, b, v)

    assert (dpnp.asnumpy(b) == v).all()


def test_multilevel_3d_strided_dpnp_array_in_kernel():
    """
    Tests if we can correctly handle a multilevel strided 3d dpnp array
    inside dpex kernel.
    """
    N = 128
    out, _, _ = dpnp.mgrid[0 : N * 2, 0 : N * 2, 0 : N * 2]  # noqa: E203

    K = 7
    b = out
    for _ in range(K):
        b = b[::2, ::2, ::2]

    r = Range(
        int(N / (2 * (K - 1))), int(N / (2 * (K - 1))), int(N / (2 * (K - 1)))
    )
    v = -3
    exp_dpex.call_kernel(change_values_3d, r, b, v)

    assert (dpnp.asnumpy(b) == v).all()
