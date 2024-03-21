# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.kernel_api as kapi


@pytest.fixture(params=[dpnp.ones(10), np.ones(10)])
def ref_type_options(request):
    return request.param


def test_atomic_ref_compilation():
    @dpex.kernel
    def atomic_ref_kernel(item: kapi.Item, a, b):
        i = item.get_id(0)
        v = kapi.AtomicRef(b, index=0)
        v.fetch_add(a[i])

    a = dpnp.ones(10)
    b = dpnp.zeros(10)
    try:
        dpex.call_kernel(atomic_ref_kernel, kapi.Range(10), a, b)
    except Exception:
        pytest.fail("Unexpected execution failure")


def test_atomic_ref_3_dim_compilation():
    @dpex.kernel
    def atomic_ref_kernel(item: kapi.Item, a, b):
        i = item.get_id(0)
        v = kapi.AtomicRef(b, index=(1, 1, 1))
        v.fetch_add(a[i])

    a = dpnp.ones(8)
    b = dpnp.zeros((2, 2, 2))

    want = np.zeros((2, 2, 2))
    want[1, 1, 1] = a.size

    try:
        dpex.call_kernel(atomic_ref_kernel, kapi.Range(a.size), a, b)
    except Exception:
        pytest.fail("Unexpected execution failure")

    assert np.array_equal(b.asnumpy(), want)


def test_atomic_ref_compilation_failure():
    """A negative test that verifies that a TypingError is raised if we try to
    create an AtomicRef in the local address space from a global address space
    ref.
    """

    @dpex.kernel
    def atomic_ref_kernel(item: kapi.Item, a, b):
        i = item.get_id(0)
        v = kapi.AtomicRef(b, index=0, address_space=kapi.AddressSpace.LOCAL)
        v.fetch_add(a[i])

    a = dpnp.ones(10)
    b = dpnp.zeros(10)

    with pytest.raises(TypingError):
        dpex.call_kernel(atomic_ref_kernel, kapi.Range(10), a, b)


def test_atomic_ref_compilation_local_accessor():
    """Tests if an AtomicRef object can be constructed from a LocalAccessor"""

    @dpex.kernel
    def atomic_ref_slm_kernel(nditem: kapi.Item, a, slm):
        gi = nditem.get_global_id(0)
        v = kapi.AtomicRef(slm, 0)
        v.fetch_add(a.dtype.type(5))
        gr = nditem.get_group()
        kapi.group_barrier(gr)
        a[gi] = slm[0]

    a = dpnp.zeros(32)
    slm = kapi.LocalAccessor(1, a.dtype)
    dpex.call_kernel(atomic_ref_slm_kernel, kapi.NdRange((32,), (32,)), a, slm)
    want = dpnp.full_like(a, 32 * a.dtype.type(5))
    assert np.allclose(a.asnumpy(), want.asnumpy())


def test_atomic_ref_creation(ref_type_options):
    """Tests AtomicRef construction from supported array types."""
    try:
        kapi.AtomicRef(ref_type_options, 0)
    except:
        pytest.fail("Unexpected error in creating an AtomicRef")


def test_atomic_ref_creation_expected_failure():
    """Tests AtomicRef construction failure from unsupported types."""

    with pytest.raises(TypeError):
        kapi.AtomicRef(1, 0)
