# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dpnp.empty overload."""

import dpctl
import dpnp
import pytest
from numba import errors

from numba_dpex import dpjit

shapes = [11, (2, 5)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_from_device(shape, dtype, usm_type):
    device = dpctl.SyclDevice().filter_string

    @dpjit
    def func(shape):
        c = dpnp.empty(shape, dtype=dtype, usm_type=usm_type, device=device)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.empty inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_empty_from_queue(shape, dtype, usm_type):
    @dpjit
    def func(shape, queue):
        c = dpnp.empty(shape, dtype=dtype, usm_type=usm_type, sycl_queue=queue)
        return c

    queue = dpctl.SyclQueue()

    try:
        c = func(shape, queue)
    except Exception:
        pytest.fail("Calling dpnp.empty inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == queue.sycl_device.filter_string

    if c.sycl_queue != queue:
        pytest.xfail("Returned queue does not have the queue passed to empty.")


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_empty_default(shape):
    @dpjit
    def func(shape):
        c = dpnp.empty(shape)
        return c

    try:
        c = func(shape)
    except Exception:
        pytest.fail("Calling dpnp.empty inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    dummy_tensor = dpctl.tensor.empty(shape)

    assert c.dtype == dummy_tensor.dtype
    assert c.usm_type == dummy_tensor.usm_type
    assert c.sycl_device == dummy_tensor.sycl_device


def test_dpnp_empty_exceptions():
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
