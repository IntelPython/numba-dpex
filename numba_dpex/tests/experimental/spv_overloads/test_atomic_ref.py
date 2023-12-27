# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import pytest
from numba.core.errors import TypingError

import numba_dpex as dpex
import numba_dpex.experimental as dpex_exp
from numba_dpex.experimental.kernel_iface import AddressSpace, AtomicRef


def test_atomic_ref_compilation():
    @dpex_exp.kernel
    def atomic_ref_kernel(a, b):
        i = dpex.get_global_id(0)
        v = AtomicRef(b, index=0, address_space=AddressSpace.GLOBAL)
        v.fetch_add(a[i])

    a = dpnp.ones(10)
    b = dpnp.zeros(10)
    try:
        dpex_exp.call_kernel(atomic_ref_kernel, dpex.Range(10), a, b)
    except Exception:
        pytest.fail("Unexpected execution failure")


def test_atomic_ref_compilation_failure():
    """A negative test that verifies that a TypingError is raised if we try to
    create an AtomicRef in the local address space from a global address space
    ref.
    """

    @dpex_exp.kernel
    def atomic_ref_kernel(a, b):
        i = dpex.get_global_id(0)
        v = AtomicRef(b, index=0, address_space=AddressSpace.LOCAL)
        v.fetch_add(a[i])

    a = dpnp.ones(10)
    b = dpnp.zeros(10)

    with pytest.raises(TypingError):
        dpex_exp.call_kernel(atomic_ref_kernel, dpex.Range(10), a, b)
