# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import math

import dpctl
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
def test_dpnp_full_default(shape, fill_value):
    """Test dpnp.full() with default parameters inside dpjit."""

    @dpjit
    def func(shape, fill_value):
        c = dpnp.full(shape, fill_value)
        return c

    try:
        c = func(shape, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    dummy = dpnp.full(shape, fill_value)

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
    assert numpy.array_equal(c.asnumpy(), dummy.asnumpy())


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_full_from_device(shape, fill_value, dtype, usm_type):
    """ "Use device only in dpnp.full() inside dpjit."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape, fill_value):
        c = dpnp.full(
            shape, fill_value, dtype=dtype, usm_type=usm_type, device=device
        )
        return c

    try:
        c = func(shape, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    if c.sycl_queue != dpctl._sycl_queue_manager.get_device_cached_queue(
        device
    ):
        pytest.xfail(
            "Returned queue does not have the queue cached against the device."
        )

    # dummy = dpnp.full(shape, fill_value, dtype=dtype)
    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full(shape, fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_full_from_queue(shape, fill_value, dtype, usm_type):
    """ "Use queue only in dpnp.full() inside dpjit."""

    @dpjit
    def func(shape, fill_value, queue):
        c = dpnp.full(
            shape, fill_value, dtype=dtype, usm_type=usm_type, sycl_queue=queue
        )
        return c

    queue = dpctl.SyclQueue()

    try:
        c = func(shape, fill_value, queue)
    except Exception:
        pytest.fail("Calling dpnp.full() inside dpjit failed.")

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

    # dummy = dpnp.full(shape, fill_value, dtype=dtype)
    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full(shape, fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)


def test_dpnp_full_exceptions():
    """Test if exception is raised when both queue and device are specified."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape, fill_value, queue):
        c = dpnp.ones(shape, fill_value, sycl_queue=queue, device=device)
        return c

    queue = dpctl.SyclQueue()

    try:
        func(10, 7, queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert "`device` and `sycl_queue` are exclusive keywords" in str(e)
