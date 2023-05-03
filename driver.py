# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for checking enforcing CFD in parfor pass.
"""

import dpctl
import dpnp
import numba as nb

import numba_dpex as dpex


@dpex.dpjit
def vecadd_prange(a, b):
    c = dpnp.empty(a.shape, dtype=a.dtype)
    # for idx in nb.prange(a.size):
    #     c[idx] = a[idx] + b[idx]
    return c


a = dpnp.ones(10)
b = dpnp.ones(10)


c = vecadd_prange(a, b)
print(c)
