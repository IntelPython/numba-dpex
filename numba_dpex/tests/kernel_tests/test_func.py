# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import numpy

import numba_dpex as dpex


@dpex.func
def g(a):
    return a + 1


@dpex.kernel
def f(a, b):
    i = dpex.get_global_id(0)
    b[i] = g(a[i])


def test_func_call_from_kernel():
    a = dpnp.ones(1024)
    b = dpnp.ones(1024)

    dpex.call_kernel(f, dpex.Range(1024), a, b)
    nb = dpnp.asnumpy(b)
    assert numpy.all(nb == 2)
