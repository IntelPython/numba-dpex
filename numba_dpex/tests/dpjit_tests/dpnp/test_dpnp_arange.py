# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp.arange() constructor."""

import dpctl
import dpnp
import numpy as np
import pytest
from numba import errors

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes


def get_xfail_test(param, reason):
    return pytest.param(
        param,
        marks=pytest.mark.xfail(reason=reason),
    )


dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=False, no_complex=True
)
dtypes_except_none = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)
usm_types = ["device", "shared", "host"]
ranges = [
    [1, None, None],  # 0
    [1, 10, None],
    [1, 10, 1],
    [-10, -1, 1],
    [11, 41, 7],
    [1, 10, 1.0],  # 5
    [1, 10.0, 1],
    [0.7, 0.91, 0.03],
    [-1003.345, -987.44, 0.73],
    [1.15, 2.75, 0.05],
    [0.75, 10.23, 0.95],  # 10
    [10.23, 0.75, -0.95],
    get_xfail_test([-1.0, None, None], "Can't allocate an empty array"),
    get_xfail_test([-1.0, 10, -2], "Impossible range"),
    get_xfail_test([-10, -1, -1], "Impossible range"),
]


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
def test_dpnp_arange_default(range, dtype):
    start, stop, step = range

    @dpjit
    def func():
        x = dpnp.arange(start, stop=stop, step=step, dtype=dtype)
        return x

    try:
        c = func()
    except Exception:
        pytest.fail("Calling dpnp.arange() inside dpjit failed.")

    a = dpnp.arange(
        start,
        stop=stop,
        step=step,
        dtype=dtype,
    )

    assert a.dtype == c.dtype
    assert a.shape == c.shape
    if a.dtype in [dpnp.float, dpnp.float16, dpnp.float32, dpnp.float64]:
        assert np.allclose(a.asnumpy(), c.asnumpy())
    else:
        assert np.array_equal(a.asnumpy(), c.asnumpy())
    if c.sycl_queue != a.sycl_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as in the dummy array."
        )
    assert c.sycl_queue == dpctl._sycl_queue_manager.get_device_cached_queue(
        a.sycl_device
    )


@pytest.mark.parametrize("range", ranges[0:3])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_arange_from_device(range, dtype, usm_type):
    device = dpctl.SyclDevice().filter_string

    start, stop, step = range

    @dpjit
    def func():
        x = dpnp.arange(
            start,
            stop=stop,
            step=step,
            dtype=dtype,
            usm_type=usm_type,
            device=device,
        )
        return x

    try:
        c = func()
    except Exception:
        pytest.fail("Calling dpnp.arange() inside dpjit failed.")

    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    if c.sycl_queue != dpctl._sycl_queue_manager.get_device_cached_queue(
        device
    ):
        pytest.xfail(
            "Returned queue does not have the same queue as cached against the device."
        )


@pytest.mark.parametrize("range", ranges[0:3])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_arange_from_queue(range, dtype, usm_type):
    start, stop, step = range

    @dpjit
    def func(queue):
        x = dpnp.arange(
            start,
            stop=stop,
            step=step,
            dtype=dtype,
            usm_type=usm_type,
            sycl_queue=queue,
        )
        return x

    try:
        queue = dpctl.SyclQueue()
        c = func(queue)
    except Exception:
        pytest.fail("Calling dpnp.arange() inside dpjit failed.")

    assert c.usm_type == usm_type
    assert c.sycl_device == queue.sycl_device
    if c.sycl_queue != queue:
        pytest.xfail(
            "Returned queue does not have the same queue as the one passed to the dpnp function."
        )
