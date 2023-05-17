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
def test_dpnp_full_like_default(shape, fill_value):
    """Test dpnp.full_like() with default parameters inside dpjit."""

    @dpjit
    def func(x, fill_value):
        y = dpnp.full_like(x, fill_value)
        return y

    try:
        a = dpnp.zeros(shape)
        c = func(a, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == a.shape[0]
    else:
        assert c.shape == a.shape

    dummy = dpnp.full_like(a, fill_value)

    assert c.dtype == dummy.dtype
    assert c.usm_type == dummy.usm_type
    assert c.sycl_device == dummy.sycl_device
    if c.sycl_queue != dummy.sycl_queue:
        pytest.xfail(
            "Returned queue does not have the queue in the dummy array."
        )
    assert c.sycl_queue == dpctl._sycl_queue_manager.get_device_cached_queue(
        dummy.sycl_device
    )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_full_like_from_device(shape, fill_value, dtype, usm_type):
    """ "Use device only in dpnp.full_like() inside dpjit."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(x, fill_value):
        y = dpnp.full_like(
            x, fill_value, dtype=dtype, usm_type=usm_type, device=device
        )
        return y

    try:
        a = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, device=device)
        c = func(a, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == a.shape[0]
    else:
        assert c.shape == a.shape

    assert c.dtype == a.dtype
    assert c.usm_type == a.usm_type
    assert c.sycl_device.filter_string == device
    if c.sycl_queue != dpctl._sycl_queue_manager.get_device_cached_queue(
        device
    ):
        pytest.xfail(
            "Returned queue does not have the queue cached against the device."
        )

    # dummy = dpnp.full_like(a, fill_value, dtype=dtype)
    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full_like(a.asnumpy(), fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_full_like_from_queue(shape, fill_value, dtype, usm_type):
    """ "Use queue only in dpnp.full_like() inside dpjit."""

    @dpjit
    def func(x, fill_value, queue):
        y = dpnp.full_like(
            x, fill_value, dtype=dtype, usm_type=usm_type, sycl_queue=queue
        )
        return y

    queue = dpctl.SyclQueue()

    try:
        a = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        c = func(a, fill_value, queue)
    except Exception:
        pytest.fail("Calling dpnp.full_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device == queue.sycl_device

    if c.sycl_queue != queue:
        pytest.xfail(
            "Returned queue does not have the queue passed to the dpnp function."
        )

    # dummy = dpnp.full_like(a, fill_value, dtype=dtype)
    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full_like(a.asnumpy(), fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)


def test_dpnp_full_like_exceptions():
    """Test if exception is raised when both queue and device are specified."""

    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func1(x, fill_value, queue):
        y = dpnp.full_like(x, 7, sycl_queue=queue, device=device)
        return y

    queue = dpctl.SyclQueue()

    try:
        a = dpnp.zeros(10)
        func1(a, 7, queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert "`device` and `sycl_queue` are exclusive keywords" in str(e)

    @dpjit
    def func2(x, fill_value):
        y = dpnp.full_like(x, fill_value, shape=(3, 3))
        return y

    try:
        func2(a, 7)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function full_like"
            in str(e)
        )


@pytest.mark.xfail
def test_dpnp_full_like_from_numpy():
    """Test if dpnp works with numpy array (it shouldn't)"""

    @dpjit
    def func(x, fill_value):
        y = dpnp.full_like(x, fill_value)
        return y

    a = numpy.ones(10)

    with pytest.raises(Exception):
        func(a, 7)


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_full_like_from_scalar(shape):
    """Test if works with scalar argument in place of an array"""

    @dpjit
    def func(shape, fill_value):
        x = dpnp.full_like(shape, fill_value)
        return x

    try:
        func(shape, 7)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function full_like"
            in str(e)
        )
