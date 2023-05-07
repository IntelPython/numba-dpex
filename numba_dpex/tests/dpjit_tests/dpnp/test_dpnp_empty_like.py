# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dpnp.empty_like overload."""


import dpctl
import dpnp
import pytest
from numba import errors

from numba_dpex import dpjit

shapes = [10, (2, 5)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_like_from_device(shape, dtype, usm_type):
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(a):
        c = dpnp.empty_like(a, dtype=dtype, usm_type=usm_type, device=device)
        return c

    NZ = dpnp.empty(shape)

    try:
        c = func(NZ)
    except Exception:
        pytest.fail("Calling dpnp.empty_like inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == NZ.shape[0]
    else:
        assert c.shape == NZ.shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    assert c.sycl_queue == dpctl.get_device_cached_queue(device)


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_like_from_queue(shape, dtype, usm_type):
    @dpjit
    def func(a, q):
        c = dpnp.empty_like(a, dtype=dtype, usm_type=usm_type, sycl_queue=q)
        return c

    NZ = dpnp.empty(shape)
    queue = dpctl.SyclQueue()

    try:
        c = func(NZ, queue)
    except Exception:
        pytest.fail("Calling dpnp.empty_like inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == NZ.shape[0]
    else:
        assert c.shape == NZ.shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_queue == NZ.sycl_queue
    assert c.sycl_queue != queue


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_empty_like_default(shape):
    @dpjit
    def func(arr):
        c = dpnp.empty_like(arr)
        return c

    arr = dpnp.empty(shape)
    try:
        c = func(arr)
    except Exception:
        pytest.fail("Calling dpnp.empty_like inside dpjit failed")

    assert c.shape == arr.shape
    assert c.dtype == arr.dtype
    assert c.usm_type == arr.usm_type
    assert c.sycl_queue == arr.sycl_queue


@pytest.mark.xfail
def test_dpnp_empty_like_from_freevar_queue():
    queue = dpctl.SyclQueue()

    @dpjit
    def func():
        c = dpnp.empty_like(10, sycl_queue=queue)
        return c

    try:
        func()
    except Exception:
        pytest.fail("Calling dpnp.empty_like inside dpjit failed")


def test_dpnp_empty_like_exceptions():
    @dpjit
    def func1(a):
        c = dpnp.empty_like(a, shape=(3, 3))
        return c

    try:
        func1(dpnp.empty((5, 5)))
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert (
            "No implementation of function Function(<function empty_like"
            in str(e)
        )

    queue = dpctl.SyclQueue()

    @dpjit
    def func2(a, q):
        c = dpnp.empty_like(a, sycl_queue=q, device="cpu")
        return c

    try:
        func2(dpnp.empty((5, 5)), queue)
    except Exception as e:
        assert isinstance(e, errors.TypingError)
        assert "`device` and `sycl_queue` are exclusive keywords" in str(e)
