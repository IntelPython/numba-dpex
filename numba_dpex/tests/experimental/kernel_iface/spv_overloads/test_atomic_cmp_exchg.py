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

list_of_cmp_exchg_funcs = [
    "compare_exchange_weak",
    "compare_exchange_strong",
]


@pytest.fixture(params=list_of_cmp_exchg_funcs)
def cmp_exchg_fn(request):
    return request.param


@pytest.fixture(params=list_of_supported_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 10
    a = dpnp.zeros(2 * N, dtype=request.param)
    b = dpnp.arange(N, dtype=request.param)
    return a, b


def test_load_store_fn(input_arrays):
    """A test for load/store atomic functions."""

    @dpex_exp.kernel
    def _kernel(a, b):
        i = dpex.get_global_id(0)
        a_ref = AtomicRef(a, index=i)
        b_ref = AtomicRef(b, index=i)
        a_ref.store(b_ref.load())

    a, b = input_arrays

    dpex_exp.call_kernel(_kernel, dpex.Range(b.size), a, b)
    # Verify that `b[i]` loaded and stored into a[i] by kernel
    # matches the `b[i]` loaded stored into a[i] using Python
    for i in range(b.size):
        a_ref = AtomicRef(a, index=i + b.size)
        b_ref = AtomicRef(b, index=i)
        a_ref.store(b_ref.load())

    for i in range(b.size):
        assert a[i] == a[i + b.size]


def test_exchange_fn(input_arrays):
    """A test for exchange atomic function."""

    @dpex_exp.kernel
    def _kernel(a, b):
        i = dpex.get_global_id(0)
        v = AtomicRef(a, index=i)
        b[i] = v.exchange(b[i])

    a_orig, b_orig = input_arrays
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


@pytest.fixture(params=["store", "exchange"])
def store_exchange_fn(request):
    return request.param


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
