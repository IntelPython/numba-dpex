#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import numpy as np
import pytest
from numba import njit

from numba_dpex import dpjit, prange

from ._helper import decorators


@pytest.mark.parametrize("jit", decorators)
def test_one_prange_mul(jit):
    @jit
    def f(a, b):
        for i in prange(4):
            b[i, 0] = a[i, 0] * 10
        return

    device = dpctl.select_default_device()

    m = 8
    n = 8
    a = dpnp.ones((m, n), device=device)
    b = dpnp.ones((m, n), device=device)

    f(a, b)
    na = dpnp.asnumpy(a)
    nb = dpnp.asnumpy(b)

    for i in range(4):
        assert nb[i, 0] == na[i, 0] * 10


@pytest.mark.parametrize("jit", decorators)
def test_one_prange_mul_nested(jit):
    @jit
    def f_inner(a, b):
        for i in prange(4):
            b[i, 0] = a[i, 0] * 10
        return

    @jit
    def f(a, b):
        return f_inner(a, b)

    device = dpctl.select_default_device()

    m = 8
    n = 8
    a = dpnp.ones((m, n), device=device)
    b = dpnp.ones((m, n), device=device)

    f(a, b)
    na = dpnp.asnumpy(a)
    nb = dpnp.asnumpy(b)

    for i in range(4):
        assert nb[i, 0] == na[i, 0] * 10


@pytest.mark.skip(reason="dpnp.add() doesn't support variable + scalar.")
def test_one_prange_add_scalar():
    @dpjit
    def f(a, b):
        for i in prange(4):
            b[i, 0] = a[i, 0] + 10
        return

    device = dpctl.select_default_device()

    m = 8
    n = 8
    a = dpnp.ones((m, n), device=device)
    b = dpnp.ones((m, n), device=device)

    f(a, b)

    for i in range(4):
        assert b[i, 0] == a[i, 0] + 10


@pytest.mark.skip(reason="[i,:] like indexing is not supported yet.")
def test_prange_2d_array():
    device = dpctl.select_default_device()
    n = 10

    @dpjit
    def f(a, b, c):
        for i in prange(n):
            c[i, :] = a[i, :] + b[i, :]
        return

    a = dpnp.ones((n, n), dtype=dpnp.int32, device=device)
    b = dpnp.ones((n, n), dtype=dpnp.int32, device=device)
    c = dpnp.ones((n, n), dtype=dpnp.int32, device=device)

    f(a, b, c)

    np.testing.assert_equal(c.asnumpy(), np.ones((n, n), dtype=np.int32) * 2)


@pytest.mark.skip(reason="Nested prange is not supported yet.")
def test_nested_prange():
    @dpjit
    def f(a, b):
        # dimensions must be provided as scalar
        m, n = a.shape
        for i in prange(m):
            for j in prange(n):
                b[i, j] = a[i, j] * 10
        return

    device = dpctl.select_default_device()

    m = 8
    n = 8
    a = dpnp.ones((m, n), device=device)
    b = dpnp.ones((m, n), device=device)

    f(a, b)

    assert np.all(b.asnumpy() == 10)


@pytest.mark.skip(reason="Nested prange is not supported yet.")
def test_multiple_prange():
    @dpjit
    def f(a, b):
        # dimensions must be provided as scalar
        m, n = a.shape
        for i in prange(m):
            val = 10
            for j in prange(n):
                b[i, j] = a[i, j] * val

        for i in prange(m):
            for j in prange(n):
                a[i, j] = a[i, j] * 10
        return

    device = dpctl.select_default_device()

    m = 8
    n = 8
    a = dpnp.ones((m, n), device=device)
    b = dpnp.ones((m, n), device=device)

    f(a, b)

    assert np.all(b.asnumpy() == 10)
    assert np.all(a.asnumpy() == 10)


@pytest.mark.skip(reason="Nested prange is not supported yet.")
def test_three_prange():
    @dpjit
    def f(a, b):
        # dimensions must be provided as scalar
        m, n, o = a.shape
        for i in prange(m):
            val = 10
            for j in prange(n):
                constant = 2
                for k in prange(o):
                    b[i, j, k] = a[i, j, k] * (val + constant)
        return

    device = dpctl.select_default_device()

    m = 8
    n = 8
    o = 8
    a = dpnp.ones((m, n, o), device=device)
    b = dpnp.ones((m, n, o), device=device)

    f(a, b)

    assert np.all(b.asnumpy() == 12)


@pytest.mark.parametrize("jit", decorators)
def test_two_consecutive_prange(jit):
    @jit
    def prange_example(a, b, c, d):
        for i in prange(n):
            c[i] = a[i] + b[i]
        for i in prange(n):
            d[i] = a[i] - b[i]
        return

    device = dpctl.select_default_device()

    n = 10
    a = dpnp.ones((n), dtype=dpnp.float32, device=device)
    b = dpnp.ones((n), dtype=dpnp.float32, device=device)
    c = dpnp.zeros((n), dtype=dpnp.float32, device=device)
    d = dpnp.zeros((n), dtype=dpnp.float32, device=device)

    prange_example(a, b, c, d)

    np.testing.assert_equal(c.asnumpy(), np.ones((n), dtype=np.float32) * 2)
    np.testing.assert_equal(d.asnumpy(), np.zeros((n), dtype=np.float32))


@pytest.mark.skip(reason="[i,:] like indexing is not supported yet.")
def test_two_consecutive_prange_2d():
    @dpjit
    def prange_example(a, b, c, d):
        for i in prange(n):
            c[i, :] = a[i, :] + b[i, :]
        for i in prange(n):
            d[i, :] = a[i, :] - b[i, :]
        return

    device = dpctl.select_default_device()

    n = 10
    a = dpnp.ones((n, n), dtype=dpnp.int32, device=device)
    b = dpnp.ones((n, n), dtype=dpnp.int32, device=device)
    c = dpnp.ones((n, n), dtype=dpnp.int32, device=device)
    d = dpnp.ones((n, n), dtype=dpnp.int32, device=device)

    prange_example(a, b, c, d)

    np.testing.assert_equal(c.asnumpy(), np.ones((n, n), dtype=np.int32) * 2)
    np.testing.assert_equal(d.asnumpy(), np.zeros((n, n), dtype=np.int32))
