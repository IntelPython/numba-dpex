# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy

from numba_dpex import kernel_api as kapi


def test_range_kernel_call1D():
    def vecadd(item: kapi.Item, a, b, c):
        idx = item.get_id(0)
        c[idx] = a[idx] + b[idx]

    a = numpy.ones(100)
    b = numpy.ones(100)
    c = numpy.empty(100)

    kapi.call_kernel(vecadd, kapi.Range(100), a, b, c)

    assert numpy.allclose(c, a + b)


def test_range_kernel_call2D():
    def vecadd(item: kapi.Item, a, b, c):
        idx = item.get_id(0)
        jdx = item.get_id(1)
        c[idx, jdx] = a[idx, jdx] + b[idx, jdx]

    a = numpy.ones((10, 10))
    b = numpy.ones((10, 10))
    c = numpy.empty((10, 10))

    kapi.call_kernel(vecadd, kapi.Range(10, 10), a, b, c)

    assert numpy.allclose(c, a + b)


def test_range_kernel_call3D():
    def vecadd(item: kapi.Item, a, b, c):
        idx = item.get_id(0)
        jdx = item.get_id(1)
        kdx = item.get_id(2)
        c[idx, jdx, kdx] = a[idx, jdx, kdx] + b[idx, jdx, kdx]

    a = numpy.ones((5, 5, 5))
    b = numpy.ones((5, 5, 5))
    c = numpy.empty((5, 5, 5))

    kapi.call_kernel(vecadd, kapi.Range(5, 5, 5), a, b, c)

    assert numpy.allclose(c, a + b)
