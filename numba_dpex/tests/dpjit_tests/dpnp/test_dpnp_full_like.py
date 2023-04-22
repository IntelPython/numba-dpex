# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import math

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy
import pytest
from numba import errors

from numba_dpex import dpjit

shapes = [11, (3, 7)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]
devices = ["cpu", None]
fill_values = [
    7,
    -7,
    7.1,
    -7.1,
    math.pi,
    math.e,
    4294967295,
    4294967295.0,
    3.4028237e38,
]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_dpnp_full_like(shape, fill_value, dtype, usm_type, device):
    @dpjit
    def func(a, v):
        c = dpnp.full_like(a, v, dtype=dtype, usm_type=usm_type, device=device)
        return c

    if isinstance(shape, int):
        NZ = numpy.random.rand(shape)
    else:
        NZ = numpy.random.rand(*shape)

    try:
        c = func(NZ, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full_like inside dpjit failed")

    C = numpy.full_like(NZ, fill_value, dtype=dtype)

    if len(c.shape) == 1:
        assert c.shape[0] == NZ.shape[0]
    else:
        assert c.shape == NZ.shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    if device is not None:
        assert (
            c.sycl_device.filter_string
            == dpctl.SyclDevice(device).filter_string
        )
    else:
        c.sycl_device.filter_string == dpctl.SyclDevice().filter_string

    assert numpy.array_equal(dpt.asnumpy(c._array_obj), C)


def test_dpnp_full_like_exceptions():
    @dpjit
    def func1(a):
        c = dpnp.full_like(a, 7, shape=(3, 3))
        return c

    try:
        func1(numpy.random.rand(5, 5))
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function full_like"
            in str(e)
        )

    queue = dpctl.SyclQueue()

    @dpjit
    def func2(a, q):
        c = dpnp.full_like(a, 7, sycl_queue=q, device="cpu")
        return c

    try:
        func2(numpy.random.rand(5, 5), queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert "`device` and `sycl_queue` are exclusive keywords" in str(e)
