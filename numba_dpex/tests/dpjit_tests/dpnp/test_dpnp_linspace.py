# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp.linspace() constructor."""

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
dtypes_no_none = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)
usm_types = ["device", "shared", "host"]
endpoints = [True, False]
ranges = [
    get_xfail_test([None, 10, 10], "'None' type argument is invalid"),  # 0
    get_xfail_test([1, None, 10], "'None' type argument is invalid"),
    get_xfail_test([1, 10, None], "'None' type argument is invalid"),
    get_xfail_test([0.0, 0.5, 0.01], "'num' must be an int"),
    [0, 10, 9],
    [0, 1, 10],  # 5
    [0, 10, 13],
    [0, 1, 17],
    [1, 0, 17],
    [-1, -1, 10],
    [0.0, 0.5, 23],  # 10
    # [-0.5, 0.0, 10] # noqa: E800
]


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("endpoint", endpoints)
def test_dpnp_linspace_default(range, dtype, endpoint):
    start, stop, num = range

    @dpjit
    def func():
        x = dpnp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)
        return x

    try:
        c = func()
    except Exception:
        pytest.fail("Calling dpnp.linspace() inside dpjit failed.")

    a = dpnp.linspace(start, stop, num, dtype=dtype, endpoint=endpoint)

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


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("endpoint", endpoints)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_linspace_from_device(range, dtype, endpoint, usm_type):
    device = dpctl.SyclDevice().filter_string

    start, stop, num = range

    @dpjit
    def func():
        x = dpnp.linspace(
            start,
            stop,
            num,
            dtype=dtype,
            endpoint=endpoint,
            usm_type=usm_type,
            device=device,
        )
        return x

    try:
        c = func()
    except Exception:
        pytest.fail("Calling dpnp.linspace() inside dpjit failed.")

    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    if c.sycl_queue != dpctl._sycl_queue_manager.get_device_cached_queue(
        device
    ):
        pytest.xfail(
            "Returned queue does not have the same queue as cached against the device."
        )


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("endpoint", endpoints)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_linspace_from_queue(range, dtype, usm_type, endpoint):
    start, stop, num = range

    @dpjit
    def func(queue):
        x = dpnp.linspace(
            start,
            stop,
            num,
            dtype=dtype,
            endpoint=endpoint,
            usm_type=usm_type,
            sycl_queue=queue,
        )
        return x

    try:
        queue = dpctl.SyclQueue()
        c = func(queue)
    except Exception:
        pytest.fail("Calling dpnp.linspace() inside dpjit failed.")

    assert c.usm_type == usm_type
    assert c.sycl_device == queue.sycl_device
    if c.sycl_queue != queue:
        pytest.xfail(
            "Returned queue does not have the same queue as the one passed to the dpnp function."
        )
