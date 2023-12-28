# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex import float32, int32

single_signature = dpex.func(int32(int32))
list_signature = dpex.func([int32(int32), float32(float32)])

# Array size
N = 10


def increment(a):
    return a + 1


def test_basic():
    """Basic test with device func"""

    f = dpex.func(increment)

    def kernel_function(a, b):
        """Kernel function that applies f() in parallel"""
        i = dpex.get_global_id(0)
        b[i] = f(a[i])

    k = dpex.kernel(kernel_function)

    a = dpnp.ones(N)
    b = dpnp.ones(N)

    dpex.call_kernel(k, dpex.Range(N), a, b)

    assert np.array_equal(dpnp.asnumpy(b), dpnp.asnumpy(a) + 1)


def test_single_signature():
    """Basic test with single signature"""

    fi32 = single_signature(increment)

    def kernel_function(a, b):
        """Kernel function that applies fi32() in parallel"""
        i = dpex.get_global_id(0)
        b[i] = fi32(a[i])

    k = dpex.kernel(kernel_function)

    # Test with int32, should work
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.ones(N, dtype=dpnp.int32)

    dpex.call_kernel(k, dpex.Range(N), a, b)

    assert np.array_equal(dpnp.asnumpy(b), dpnp.asnumpy(a) + 1)

    # Test with int64, should fail
    a = dpnp.ones(N, dtype=dpnp.int64)
    b = dpnp.ones(N, dtype=dpnp.int64)

    with pytest.raises(Exception) as e:
        dpex.call_kernel(k, dpex.Range(N), a, b)

    assert " >>> <unknown function>(int64)" in e.value.args[0]


def test_list_signature():
    """Basic test with list signature"""

    fi32f32 = list_signature(increment)

    def kernel_function(a, b):
        """Kernel function that applies fi32f32() in parallel"""
        i = dpex.get_global_id(0)
        b[i] = fi32f32(a[i])

    k = dpex.kernel(kernel_function)

    # Test with int32, should work
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.ones(N, dtype=dpnp.int32)

    dpex.call_kernel(k, dpex.Range(N), a, b)

    assert np.array_equal(dpnp.asnumpy(b), dpnp.asnumpy(a) + 1)

    # Test with float32, should work
    a = dpnp.ones(N, dtype=dpnp.float32)
    b = dpnp.ones(N, dtype=dpnp.float32)

    dpex.call_kernel(k, dpex.Range(N), a, b)

    assert np.array_equal(dpnp.asnumpy(b), dpnp.asnumpy(a) + 1)

    # Test with int64, should fail
    a = dpnp.ones(N, dtype=dpnp.int64)
    b = dpnp.ones(N, dtype=dpnp.int64)

    with pytest.raises(Exception) as e:
        dpex.call_kernel(k, dpex.Range(N), a, b)

    assert " >>> <unknown function>(int64)" in e.value.args[0]
