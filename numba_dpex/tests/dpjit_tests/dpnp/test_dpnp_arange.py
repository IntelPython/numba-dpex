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
usm_types = ["device", "shared", "host"]
ranges = [
    [1, None, None],
    [1, 10, None],
    [1, 10, 1],
    [-10, -1, 1],
    [11, 41, 7],
    [1, 10, 1.0],
    [1, 10.0, 1],
    [0.7, 0.91, 0.03],
    [-1003.345, -987.44, 0.73],
    get_xfail_test([-1.0, None, None], "can't allocate an empty array"),
    get_xfail_test([-1.0, 10, -2], "impossible range"),
    get_xfail_test([-10, -1, -1], "impossible range"),
]


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_arange_basic(range, dtype, usm_type):
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

    a = dpnp.arange(
        start,
        stop=stop,
        step=step,
        dtype=dtype,
        usm_type=usm_type,
        device=device,
    )

    print(a)
    print(c)

    assert a.dtype == c.dtype
    if a.dtype in [dpnp.float, dpnp.float16, dpnp.float32, dpnp.float64]:
        assert np.allclose(a.asnumpy(), c.asnumpy())
    else:
        assert np.array_equal(a.asnumpy(), c.asnumpy())
    assert c.usm_type == usm_type
    assert c.sycl_device.filter_string == device
    if c.sycl_queue != dpctl._sycl_queue_manager.get_device_cached_queue(
        device
    ):
        pytest.xfail(
            "Returned queue does not have the same queue as cached against the device."
        )
