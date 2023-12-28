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


@dpex_exp.kernel
def atomic_ref_0(a):
    i = dpex.get_global_id(0)
    v = AtomicRef(a, index=0)
    v.fetch_add(a[i + 2])


@dpex_exp.kernel
def atomic_ref_1(a):
    i = dpex.get_global_id(0)
    v = AtomicRef(a, index=1)
    v.fetch_add(a[i + 2])


def test_spirv_compiler_flags():
    """Check if float atomic flag is being populated from intrinsic for the
    second call.

    https://github.com/IntelPython/numba-dpex/issues/1262
    """
    N = 10
    a = dpnp.ones(N, dtype=dpnp.float32)

    dpex_exp.call_kernel(atomic_ref_0, dpex.Range(N - 2), a)
    dpex_exp.call_kernel(atomic_ref_1, dpex.Range(N - 2), a)

    assert a[0] == N - 1
    assert a[1] == N - 1
