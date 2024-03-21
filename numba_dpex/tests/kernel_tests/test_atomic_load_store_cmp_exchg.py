# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
from numba_dpex.kernel_api import AtomicRef
from numba_dpex.tests._helper import get_all_dtypes

list_of_supported_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)


@pytest.fixture(params=["store", "exchange"])
def store_exchange_fn(request):
    return request.param


@pytest.mark.parametrize("supported_dtype", list_of_supported_dtypes)
def test_load_store_fn(supported_dtype):
    """A test for load/store atomic functions."""

    @dpex.kernel
    def _kernel(item, a, b):
        i = item.get_id(0)
        a_ref = AtomicRef(a, index=i)
        b_ref = AtomicRef(b, index=i)
        val = b_ref.load()
        a_ref.store(val)

    N = 10
    a = dpnp.zeros(2 * N, dtype=supported_dtype)
    b = dpnp.arange(N, dtype=supported_dtype)

    dpex.call_kernel(_kernel, dpex.Range(b.size), a, b)
    # Verify that `b[i]` loaded and stored into a[i] by kernel
    # matches the `b[i]` loaded stored into a[i] using Python
    for i in range(b.size):
        a_ref = AtomicRef(a, index=i + b.size)
        b_ref = AtomicRef(b, index=i)
        a_ref.store(b_ref.load())

    for i in range(b.size):
        assert a[i] == a[i + b.size]


@pytest.mark.parametrize("supported_dtype", list_of_supported_dtypes)
def test_exchange_fn(supported_dtype):
    """A test for exchange atomic function."""

    @dpex.kernel
    def _kernel(item, a, b):
        i = item.get_id(0)
        v = AtomicRef(a, index=i)
        b[i] = v.exchange(b[i])

    N = 10
    a_orig = dpnp.zeros(2 * N, dtype=supported_dtype)
    b_orig = dpnp.arange(N, dtype=supported_dtype)

    a_copy = dpnp.copy(a_orig)
    b_copy = dpnp.copy(b_orig)

    dpex.call_kernel(_kernel, dpex.Range(b_orig.size), a_copy, b_copy)

    # Values in `b` have been exchanged
    # with values in `a`.
    # Test if `a_copy` is same as `b_orig`
    # and `b_copy` is same as `a_orig`
    for i in range(b_orig.size):
        assert a_copy[i] == b_orig[i]
        assert b_copy[i] == a_orig[i]


@pytest.mark.parametrize("supported_dtype", list_of_supported_dtypes)
def test_compare_exchange_fns(supported_dtype):
    """A test for compare exchange atomic functions."""

    @dpex.kernel
    def _kernel(b):
        b_ref = AtomicRef(b, index=1)
        b[0] = b_ref.compare_exchange(
            expected_ref=b, desired=b[3], expected_idx=2
        )

    b = dpnp.arange(4, dtype=supported_dtype)

    dpex.call_kernel(_kernel, dpex.Range(1), b)

    # check for failure
    assert b[0] == 0
    assert b[2] == b[1]

    dpex.call_kernel(_kernel, dpex.Range(1), b)

    # check for success
    assert b[0] == 1
    assert b[1] == b[3]


def test_store_exchange_diff_types(store_exchange_fn):
    """A negative test that verifies that a TypingError is raised if
    AtomicRef type and value are of different types.
    """

    @dpex.kernel
    def _kernel(item, a, b):
        i = item.get_id(0)
        v = AtomicRef(b, index=0)
        getattr(v, store_exchange_fn)(a[i])

    N = 10
    a = dpnp.ones(N, dtype=dpnp.float32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    with pytest.raises(TypingError):
        dpex.call_kernel(_kernel, dpex.Range(10), a, b)
