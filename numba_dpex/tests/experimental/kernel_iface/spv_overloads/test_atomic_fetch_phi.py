# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import pytest

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.experimental.kernel_iface import AtomicRef
from numba_dpex.tests._helper import get_all_dtypes

list_of_supported_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=list_of_supported_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 10
    a = dpnp.ones(N, dtype=request.param)
    b = dpnp.zeros(N, dtype=request.param)
    return a, b


@pytest.mark.parametrize("ref_index", [0, 5])
def test_fetch_add(input_arrays, ref_index):
    @dpex_exp.kernel
    def atomic_ref_kernel(a, b, ref_index):
        i = dpex.get_global_id(0)
        v = AtomicRef(b, index=ref_index)
        v.fetch_add(a[i])

    a, b = input_arrays

    dpex_exp.call_kernel(atomic_ref_kernel, dpex.Range(10), a, b, ref_index)

    # Verify that `a` was accumulated at b[ref_index]
    assert b[ref_index] == 10
