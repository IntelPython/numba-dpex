# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

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
devices = ["cpu", "unknown"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_dpnp_zeros_like(shape, dtype, usm_type, device):
    @dpjit
    def func(a):
        c = dpnp.zeros_like(a, dtype=dtype, usm_type=usm_type, device=device)
        return c

    if isinstance(shape, int):
        NZ = numpy.random.rand(shape)
    else:
        NZ = numpy.random.rand(*shape)

    try:
        c = func(NZ)
    except Exception:
        pytest.fail("Calling dpnp.zeros_like inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == NZ.shape[0]
    else:
        assert c.shape == NZ.shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    if device != "unknown":
        assert (
            c.sycl_device.filter_string
            == dpctl.SyclDevice(device).filter_string
        )
    else:
        c.sycl_device.filter_string == dpctl.SyclDevice().filter_string

    assert numpy.array_equal(
        dpt.asnumpy(c._array_obj), numpy.zeros_like(c._array_obj)
    )


def test_dpnp_zeros_like_exceptions():
    @dpjit
    def func1(a):
        c = dpnp.zeros_like(a, shape=(3, 3))
        return c

    try:
        func1(numpy.random.rand(5, 5))
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function zeros_like"
            in str(e)
        )

    queue = dpctl.SyclQueue()

    @dpjit
    def func2(a):
        c = dpnp.zeros_like(a, sycl_queue=queue)
        return c

    try:
        func2(numpy.random.rand(5, 5))
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function zeros_like"
            in str(e)
        )
