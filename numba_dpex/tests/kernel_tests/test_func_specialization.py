# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numpy as np
from numba import int32, int64

import numba_dpex as dpex

i32_signature = dpex.device_func(int32(int32))
i32i64_signature = dpex.device_func([int32(int32), int64(int64)])

# Array size
N = 1024


def increment(a):
    return a + 1


fi32 = i32_signature(increment)
fi32i64 = i32i64_signature(increment)


@dpex.kernel
def kernel_function(item, a, b):
    """Kernel function that calls fi32()"""
    i = item.get_id(0)
    b[i] = fi32(a[i])


@dpex.kernel
def kernel_function2(item, a, b):
    """Kernel function that calls  fi32i64()"""
    i = item.get_id(0)
    b[i] = fi32i64(a[i])


def test_calling_specialized_device_func():
    """Tests if a specialized device_func gets called as expected from kernel"""
    a = dpnp.ones(N, dtype=dpnp.int32)
    b = dpnp.zeros(N, dtype=dpnp.int32)

    dpex.call_kernel(kernel_function, dpex.Range(N), a, b)

    assert np.array_equal(dpnp.asnumpy(b), dpnp.asnumpy(a) + 1)


def test_calling_specialized_device_func_wrong_signature():
    """Tests that calling specialized signature with wrong signature does not
    trigger recompilation.

    Tests kernel_function with float32. Numba will downcast float32 to int32
    and call the specialized function. The implicit casting is a problem, but
    for the purpose of this test case, all we care is to check if the
    specialized function was called and we did not recompiled the device_func.
    Refer: https://github.com/numba/numba/issues/9506

    """
    # Test with int64, should fail
    a = dpnp.full(N, 1.5, dtype=dpnp.float32)
    b = dpnp.zeros(N, dtype=dpnp.float32)

    dpex.call_kernel(kernel_function, dpex.Range(N), a, b)

    # Since Numba is calling the i32 specialization of increment, the values in
    # `a` are first down converted to int32, *i.e.*, 1.5 to 1 and then
    # incremented. Thus, the output is 2 instead of 2.5.
    # The implicit down casting is a dangerous thing for Numba to do, but we use
    # to our advantage to test if re compilation did not happen for a
    # specialized device function.
    assert np.all(dpnp.asnumpy(b) == 2)
    assert not np.all(dpnp.asnumpy(b) == 2.5)


def test_multi_specialized_device_func():
    """Tests if a device_func with multiple specialization can be called
    in a kernel
    """
    # Test with int32, i64 should work
    ai32 = dpnp.ones(N, dtype=dpnp.int32)
    bi32 = dpnp.ones(N, dtype=dpnp.int32)
    ai64 = dpnp.ones(N, dtype=dpnp.int64)
    bi64 = dpnp.ones(N, dtype=dpnp.int64)

    dpex.call_kernel(kernel_function2, dpex.Range(N), ai32, bi32)
    dpex.call_kernel(kernel_function2, dpex.Range(N), ai64, bi64)

    assert np.array_equal(dpnp.asnumpy(bi32), dpnp.asnumpy(ai32) + 1)
    assert np.array_equal(dpnp.asnumpy(bi64), dpnp.asnumpy(ai64) + 1)
