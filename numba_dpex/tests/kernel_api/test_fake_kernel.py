# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numpy.testing import assert_equal

import numba_dpex as ndpx
import numba_dpex.experimental as ndpx_ex


def test_simple1():
    def func(a, b, c):
        i = ndpx.get_global_id(0)
        j = ndpx.get_global_id(1)
        k = ndpx.get_global_id(2)
        c[i, j, k] = a[i, j, k] + b[i, j, k]

    a = dpnp.array([[[1, 2, 3], [4, 5, 6]]], dpnp.int64)
    b = dpnp.array([[[7, 8, 9], [10, 11, 12]]], dpnp.int64)
    c = dpnp.zeros(a.shape, dtype=a.dtype)
    c_sim = dpnp.zeros(a.shape, dtype=a.dtype)

    # Call sim kernel
    ndpx_ex.call_kernel(func, ndpx.Range(*a.shape), a, b, c_sim)

    # Make kernel
    kfunc = ndpx_ex.kernel(func)
    # Call kernel
    ndpx_ex.call_kernel(kfunc, ndpx.Range(*a.shape), a, b, c)

    c_exp = a + b

    assert_equal(c_sim.asnumpy(), c.asnumpy())
    assert_equal(c_sim.asnumpy(), c_exp.asnumpy())
