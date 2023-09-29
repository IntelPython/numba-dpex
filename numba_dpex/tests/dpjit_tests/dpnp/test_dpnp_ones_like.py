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
from numba_dpex.tests._helper import get_all_dtypes

shapes = [11, (3, 7)]
dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_ones_like_default(shape):
    """Test dpnp.ones_like() with default parameters inside dpjit."""

    @dpjit
    def func(x):
        y = dpnp.ones_like(x)
        return y

    try:
        a = dpnp.zeros(shape, dtype=dpnp.float32)
        c = func(a)
    except Exception:
        pytest.fail("Calling dpnp.ones_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == a.shape[0]
    else:
        assert c.shape == a.shape

    assert (c.asnumpy() == 1).all()

    dummy = dpnp.ones_like(a)

    assert c.dtype == dummy.dtype
    assert c.usm_type == dummy.usm_type
    assert c.sycl_device == dummy.sycl_device
    if c.sycl_queue != dummy.sycl_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as in the dummy array."
        )
    assert c.sycl_queue == dpctl._sycl_queue_manager.get_device_cached_queue(
        dummy.sycl_device
    )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_ones_like_from_device(shape, dtype, usm_type):
    """ "Use device only in dpnp.ones_like() inside dpjit."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(x):
        y = dpnp.ones_like(x, dtype=dtype, usm_type=usm_type, device=device)
        return y

    try:
        a = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, device=device)
        c = func(a)
    except Exception:
        pytest.fail("Calling dpnp.ones_like() inside dpjit failed.")

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
            "Returned queue does not have the same queue as cached against the device."
        )
    assert (c.asnumpy() == 1).all()


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_ones_like_from_queue(shape, dtype, usm_type):
    """ "Use queue only in dpnp.ones_like() inside dpjit."""

    @dpjit
    def func(x, queue):
        y = dpnp.ones_like(x, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        return y

    try:
        queue = dpctl.SyclQueue()
        a = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        c = func(a, queue)
    except Exception:
        pytest.fail("Calling dpnp.ones_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == a.shape[0]
    else:
        assert c.shape == a.shape

    assert c.dtype == a.dtype
    assert c.usm_type == a.usm_type
    assert c.sycl_device == queue.sycl_device
    assert (c.asnumpy() == 1).all()
    assert c.sycl_queue == queue
    assert c.sycl_queue == a.sycl_queue

    try:
        queue = dpctl.SyclQueue()
        a1 = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type)
        c1 = func(a1, queue)
    except Exception:
        pytest.fail("Calling dpnp.ones_like() inside dpjit failed.")

    if len(c1.shape) == 1:
        assert c1.shape[0] == a1.shape[0]
    else:
        assert c1.shape == a1.shape

    assert c1.dtype == a1.dtype
    assert c1.usm_type == a1.usm_type
    assert c1.sycl_device == queue.sycl_device
    assert (c1.asnumpy() == 1).all()
    assert c1.sycl_queue == queue
    assert c1.sycl_queue != a1.sycl_queue


def test_dpnp_ones_like_exceptions():
    """Test if exception is raised when both queue and device are specified."""

    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func1(x, queue):
        y = dpnp.ones_like(x, sycl_queue=queue, device=device)
        return y

    try:
        queue = dpctl.SyclQueue()
        a = dpnp.zeros(10, dtype=dpnp.float32)
        func1(a, queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert "`device` and `sycl_queue` are exclusive keywords" in str(e)

    @dpjit
    def func2(x):
        y = dpnp.ones_like(x, shape=(3, 3))
        return y

    try:
        func2(a)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function ones_like"
            in str(e)
        )


def test_dpnp_ones_like_from_numpy():
    """Test if dpnp works with numpy array (it shouldn't)"""

    @dpjit
    def func(x):
        y = dpnp.ones_like(x)
        return y

    a = numpy.ones(10, dtype=dpnp.float32)

    with pytest.raises(Exception):
        func(a)


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_ones_like_from_scalar(shape):
    """Test if works with scalar argument in place of an array"""

    @dpjit
    def func(shape):
        x = dpnp.ones_like(shape)
        return x

    try:
        func(shape)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function ones_like"
            in str(e)
        )
