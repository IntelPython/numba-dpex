# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.experimental.kernel_iface import AtomicRef
from numba_dpex.tests._helper import get_all_dtypes

list_of_supported_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=["store", "exchange"])
def store_exchange_fn(request):
    return request.param


def test_load_store_fn():
    """A test for load/store atomic functions."""

    @dpex_exp.kernel
    def _kernel(a, b):
        i = dpex.get_global_id(0)
        a_ref = AtomicRef(a, index=i)
        b_ref = AtomicRef(b, index=i)
        val = b_ref.load()
        a_ref.store(val)

    N = 10
    a = dpnp.zeros(2 * N, dtype=dpnp.float32)
    b = dpnp.arange(N, dtype=dpnp.float32)

    dpex_exp.call_kernel(_kernel, dpex.Range(b.size), a, b)
    # Verify that `b[i]` loaded and stored into a[i] by kernel
    # matches the `b[i]` loaded stored into a[i] using Python
    for i in range(b.size):
        a_ref = AtomicRef(a, index=i + b.size)
        b_ref = AtomicRef(b, index=i)
        a_ref.store(b_ref.load())

    for i in range(b.size):
        assert a[i] == a[i + b.size]


def test_exchange_fn():
    """A test for exchange atomic function."""

    @dpex_exp.kernel
    def _kernel(a, b):
        i = dpex.get_global_id(0)
        v = AtomicRef(a, index=i)
        b[i] = v.exchange(b[i])

    N = 10
    a_orig = dpnp.zeros(2 * N, dtype=dpnp.float32)
    b_orig = dpnp.arange(N, dtype=dpnp.float32)

    a_copy = dpnp.copy(a_orig)
    b_copy = dpnp.copy(b_orig)

    dpex_exp.call_kernel(_kernel, dpex.Range(b_orig.size), a_copy, b_copy)

    # Values in `b` have been exchanged
    # with values in `a`.
    # Test if `a_copy` is same as `b_orig`
    # and `b_copy` is same as `a_orig`
    for i in range(b_orig.size):
        assert a_copy[i] == b_orig[i]
        assert b_copy[i] == a_orig[i]


def test_store_exchange_diff_types(store_exchange_fn):
    """A negative test that verifies that a TypingError is raised if
    AtomicRef type and value are of different types.
    """

    @dpex_exp.kernel
    def _kernel(a, b):
        i = dpex.get_global_id(0)
        v = AtomicRef(b, index=0)
        getattr(v, store_exchange_fn)(a[i])

    N = 10
    a = dpnp.ones(N, dtype=dpnp.float32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    with pytest.raises(TypingError):
        dpex_exp.call_kernel(_kernel, dpex.Range(10), a, b)