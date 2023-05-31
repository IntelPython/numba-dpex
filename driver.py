# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for checking enforcing CFD in parfor pass.
"""

import dpnp

import numba_dpex as dpex


@dpex.dpjit
def f(a, b):
    for i in dpex.prange(4):
        b[i, 0] = a[i, 0] * 10
    return


@dpex.dpjit
def g(a):
    c = dpnp.empty(a.shape, dtype=a.dtype)
    return c


m = 8
n = 8
a = dpnp.ones((m, n))
b = dpnp.ones((m, n))

f(a, b)

print(b)

c = g(b)
print(c)
