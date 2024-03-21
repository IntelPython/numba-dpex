# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
from numba_dpex.kernel_api import AtomicRef, Item, Range
from numba_dpex.tests._helper import get_all_dtypes

list_of_supported_dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)

list_of_fetch_phi_funcs = [
    "fetch_add",
    "fetch_sub",
    "fetch_min",
    "fetch_max",
    "fetch_and",
    "fetch_or",
    "fetch_xor",
]


@pytest.fixture(params=list_of_fetch_phi_funcs)
def fetch_phi_fn(request):
    return request.param


@pytest.fixture(params=list_of_supported_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 10
    a = dpnp.arange(N, dtype=request.param)
    b = dpnp.ones(N, dtype=request.param)
    return a, b


@pytest.mark.parametrize("ref_index", [0, 5])
def test_fetch_phi_fn(input_arrays, ref_index, fetch_phi_fn):
    """A test for all fetch_phi atomic functions."""

    @dpex.kernel
    def _kernel(item: Item, a, b, ref_index):
        i = item.get_id(0)
        v = AtomicRef(b, index=ref_index)
        getattr(v, fetch_phi_fn)(a[i])

    a, b = input_arrays

    if (
        fetch_phi_fn in ["fetch_and", "fetch_or", "fetch_xor"]
        and issubclass(a.dtype.type, dpnp.floating)
        and issubclass(b.dtype.type, dpnp.floating)
    ):
        # fetch_and, fetch_or, fetch_xor accept only int arguments.
        # test for TypingError when float arguments are passed.
        with pytest.raises(TypingError):
            dpex.call_kernel(_kernel, Range(10), a, b, ref_index)
    else:
        dpex.call_kernel(_kernel, Range(10), a, b, ref_index)
        # Verify that `a` accumulated at b[ref_index] by kernel
        # matches the `a` accumulated at  b[ref_index+1] using Python
        for i in range(a.size):
            v = AtomicRef(b, index=ref_index + 1)
            getattr(v, fetch_phi_fn)(a[i])

        assert b[ref_index] == b[ref_index + 1]


def test_fetch_phi_retval(fetch_phi_fn):
    """A test for all fetch_phi atomic functions."""

    @dpex.kernel
    def _kernel(item: Item, a, b, c):
        i = item.get_id(0)
        v = AtomicRef(b, index=i)
        c[i] = getattr(v, fetch_phi_fn)(a[i])

    N = 10
    a = dpnp.arange(N, dtype=dpnp.int32)
    b = dpnp.ones(N, dtype=dpnp.int32)
    c = dpnp.zeros(N, dtype=dpnp.int32)
    a_copy = dpnp.copy(a)
    b_copy = dpnp.copy(b)
    c_copy = dpnp.copy(c)

    dpex.call_kernel(_kernel, Range(10), a, b, c)

    # Verify if the value returned by fetch_phi kernel
    # stored into `c` is same as the value returned
    # by fetch_phi python stored into `c_copy`
    for i in range(a.size):
        v = AtomicRef(b_copy, index=i)
        c_copy[i] = getattr(v, fetch_phi_fn)(a_copy[i])

    for i in range(a.size):
        assert c[i] == c_copy[i]


def test_fetch_phi_diff_types(fetch_phi_fn):
    """A negative test that verifies that a TypingError is raised if
    AtomicRef type and value to be added are of different types.
    """

    @dpex.kernel
    def _kernel(item: Item, a, b):
        i = item.get_id(0)
        v = AtomicRef(b, index=0)
        getattr(v, fetch_phi_fn)(a[i])

    N = 10
    a = dpnp.ones(N, dtype=dpnp.float32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    with pytest.raises(TypingError):
        dpex.call_kernel(_kernel, Range(10), a, b)


@dpex.kernel
def atomic_ref_0(item: Item, a):
    i = item.get_id(0)
    v = AtomicRef(a, index=0)
    v.fetch_add(a[i + 2])


@dpex.kernel
def atomic_ref_1(item: Item, a):
    i = item.get_id(0)
    v = AtomicRef(a, index=1)
    v.fetch_add(a[i + 2])


def test_spirv_compiler_flags_add():
    """Check if float atomic flag is being populated from intrinsic for the
    second call.

    https://github.com/IntelPython/numba-dpex/issues/1262
    """
    N = 10
    a = dpnp.ones(N, dtype=dpnp.float32)

    dpex.call_kernel(atomic_ref_0, Range(N - 2), a)
    dpex.call_kernel(atomic_ref_1, Range(N - 2), a)

    assert a[0] == N - 1
    assert a[1] == N - 1


@dpex.kernel
def atomic_max_0(item: Item, a):
    i = item.get_id(0)
    v = AtomicRef(a, index=0)
    if i != 0:
        v.fetch_max(a[i])


@dpex.kernel
def atomic_max_1(item: Item, a):
    i = item.get_id(0)
    v = AtomicRef(a, index=0)
    if i != 0:
        v.fetch_max(a[i])


def test_spirv_compiler_flags_max():
    """Check if float atomic flag is being populated from intrinsic for the
    second call.

    https://github.com/IntelPython/numba-dpex/issues/1262
    """
    N = 10
    a = dpnp.arange(N, dtype=dpnp.float32)
    b = dpnp.arange(N, dtype=dpnp.float32)

    dpex.call_kernel(atomic_max_0, Range(N), a)
    dpex.call_kernel(atomic_max_1, Range(N), b)

    assert a[0] == N - 1
    assert b[0] == N - 1
