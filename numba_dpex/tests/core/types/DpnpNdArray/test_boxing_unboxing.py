# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing for dpnp.ndarray
"""

import dpnp

from numba_dpex import dpjit


def test_boxing_unboxing():
    """Tests basic boxing and unboxing of a dpnp.ndarray object.

    Checks if we can pass in and return a dpctl.ndarray object to and
    from a dpjit decorated function.
    """

    @dpjit
    def func(a):
        return a

    a = dpnp.empty(10, dtype=dpnp.float32)
    try:
        b = func(a)
    except:
        assert False, "Failure during unbox/box of dpnp.ndarray"

    assert a.shape == b.shape
    assert a.device == b.device
    assert a.strides == b.strides
    assert a.dtype == b.dtype
    # To ensure we are returning the original array when boxing
    assert id(a) == id(b)


def test_stride_calc_at_unboxing():
    """Tests if strides were correctly computed during unboxing."""

    def _tester(a):
        return a.strides

    b = dpnp.empty((4, 16, 4), dtype=dpnp.float32)
    strides = dpjit(_tester)(b)

    # Numba computes strides as bytes
    assert list(strides) == [256, 16, 4]
