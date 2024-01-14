# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dpnp.empty overload."""

import dpctl
import dpnp
import pytest
from numba import errors

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes

shapes = [11, (2, 5)]
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_empty_default(shape):
    """Test dpnp.empty() with default parameters inside dpjit."""

    @dpjit
    def func(shape):
        c = dpnp.empty(shape)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.empty() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    dummy = dpnp.empty(shape)

    assert c.dtype == dummy.dtype
    assert c.usm_type == dummy.usm_type
    assert c.sycl_device == dummy.sycl_device
    assert c.sycl_queue == dummy.sycl_queue
    if c.sycl_queue != dummy.sycl_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as in the dummy array."
        )
    assert c.sycl_queue == dpctl._sycl_queue_manager.get_device_cached_queue(
        dummy.sycl_device
    )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    ),
)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_from_device(shape, dtype, usm_type):
    """ "Use device only in dpnp.emtpy() inside dpjit."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape):
        c = dpnp.empty(shape, dtype=dtype, usm_type=usm_type, device=device)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.empty() inside dpjit failed.")

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
            "Returned queue does not have the same queue as cached against the device."
        )


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(
        no_bool=True, no_float16=True, no_none=True, no_complex=True
    ),
)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_from_queue(shape, dtype, usm_type):
    """ "Use queue only in dpnp.emtpy() inside dpjit."""

    @dpjit
    def func(shape, queue):
        c = dpnp.empty(shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        return c

    try:
        queue = dpctl.SyclQueue()
        c = func(shape, queue)
    except Exception:
        pytest.fail("Calling dpnp.empty() inside dpjit failed.")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device == queue.sycl_device

    if c.sycl_queue != queue:
        pytest.xfail(
            "Returned queue does not have the same queue as the one passed to the dpnp function."
        )


def test_dpnp_empty_exceptions():
    """Test if exception is raised when both queue and device are specified."""
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape, queue):
        c = dpnp.empty(
            shape, sycl_queue=queue, device=device, dtype=dpnp.float32
        )
        return c

    with pytest.raises(errors.TypingError):
        queue = dpctl.SyclQueue()
        func(10, queue)


@pytest.mark.xfail(reason="dpjit allocates new dpctl.SyclQueue on boxing")
# TODO: remove test_dpnp_empty_with_dpjit_queue_temp once pass.
def test_dpnp_empty_with_dpjit_queue():
    """Test if dpnp array can be created with a queue from another array"""

    @dpjit
    def func(a):
        queue = a.sycl_queue
        return dpnp.empty(10, sycl_queue=queue)

    a = dpnp.empty(10)
    b = func(a)

    assert id(a.sycl_queue) == id(b.sycl_queue)


def test_dpnp_empty_with_dpjit_queue_temp():
    """Test if dpnp array can be created with a queue from another array"""

    @dpjit
    def func(a):
        queue = a.sycl_queue
        return dpnp.empty(10, sycl_queue=queue)

    a = dpnp.empty(10)
    b = func(a)

    assert a.sycl_queue == b.sycl_queue
