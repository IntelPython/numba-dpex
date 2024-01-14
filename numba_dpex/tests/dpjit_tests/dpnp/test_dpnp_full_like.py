# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import math
import sys

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy
import pytest
from numba import errors

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes

shapes = [11, (3, 7)]
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
        a = dpnp.zeros(shape, dtype=dpnp.float32)
        c = func(a, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full_like() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == a.shape[0]
    else:
        assert c.shape == a.shape

    dummy = dpnp.full_like(a, fill_value)

    if c.dtype != dummy.dtype:
        if sys.platform != "linux":
            pytest.xfail(
                "Default bit length is not as same as that of linux for {0:s}".format(
                    str(dummy.dtype)
                )
            )
        else:
            pytest.fail("The dtype of the returned array doesn't conform.")

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
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    ),
)
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
            "Returned queue does not have the same queue as cached against the device."
        )

    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full_like(a.asnumpy(), fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("fill_value", fill_values)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    ),
)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_full_like_from_queue(shape, fill_value, dtype, usm_type):
    """ "Use queue only in dpnp.full_like() inside dpjit."""

    @dpjit
    def func(x, fill_value, queue):
        y = dpnp.full_like(
            x, fill_value, dtype=dtype, usm_type=usm_type, sycl_queue=queue
        )
        return y

    try:
        queue = dpctl.SyclQueue()
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
    assert c.sycl_queue == a.sycl_queue
    assert c.sycl_queue == queue

    # dpnp can't cast 4294967295 into int32 and so on,
    # but we can, also numpy can, so we are using numpy here
    dummy = numpy.full_like(a.asnumpy(), fill_value, dtype=dtype)
    assert numpy.array_equal(c.asnumpy(), dummy)

    try:
        queue = dpctl.SyclQueue()
        a1 = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type)
        c1 = func(a1, fill_value, queue)

        if len(c1.shape) == 1:
            assert c1.shape[0] == shape
        else:
            assert c1.shape == shape

        assert c1.dtype == dtype
        assert c1.usm_type == usm_type
        assert c1.sycl_device == queue.sycl_device
        assert c1.sycl_queue == queue
        assert c1.sycl_queue != a1.sycl_queue

    except Exception:
        pytest.fail("Calling dpnp.full_like() inside dpjit failed.")


def test_dpnp_full_like_exceptions():
    """Test if exception is raised when both queue and device are specified."""

    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func1(x, fill_value, queue):
        y = dpnp.full_like(x, 7, sycl_queue=queue, device=device)
        return y

    try:
        queue = dpctl.SyclQueue()
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
