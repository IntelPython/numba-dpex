# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for slicing dpnp.ndarray
"""

import dpnp
import numpy

from numba_dpex import dpjit


def test_1d_slicing():
    """Tests if dpjit properly computes strides and returns them to Python."""

    def _tester(a):
        return a[1:5]

    a = dpnp.arange(10)
    b = dpnp.asnumpy(dpjit(_tester)(a))

    na = numpy.arange(10)
    nb = _tester(na)

    assert (b == nb).all()


def test_1d_slicing2():
    """Tests if dpjit properly computes strides and returns them to Python."""

    def _tester(a):
        b = a[1:4]
        a[6:9] = b

    a = dpnp.arange(10)
    b = dpnp.asnumpy(dpjit(_tester)(a))

    na = numpy.arange(10)
    nb = _tester(na)

    assert (b == nb).all()


def test_multidim_slicing():
    """Tests if dpjit properly slices strides and returns them to Python."""

    def _tester(a, b):
        b[:, :, 0] = a

    a = dpnp.arange(64, dtype=numpy.int64)
    a = a.reshape(4, 16)
    b = dpnp.empty((4, 16, 4), dtype=numpy.int64)
    dpjit(_tester)(a, b)

    na = numpy.arange(64, dtype=numpy.int64)
    na = na.reshape(4, 16)
    nb = numpy.empty((4, 16, 4), dtype=numpy.int64)
    _tester(na, nb)

    assert (nb[:, :, 0] == dpnp.asnumpy(b)[:, :, 0]).all()
