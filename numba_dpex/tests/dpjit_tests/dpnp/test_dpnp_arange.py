# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp.arange() constructor."""

import dpctl
import dpnp
import numpy as np
import pytest

from numba_dpex import dpjit
from numba_dpex.tests._helper import get_all_dtypes


def get_xfail_test(param, reason):
    """Generate an X-fail test `pytest` parameter.

    Args:
        param (list): A `list` of valid parameters.
        reason (str): A `str` describing the reason for failure.

    Returns:
        pytest.param: A `pytest.param` parameter.
    """
    return pytest.param(
        param,
        marks=pytest.mark.xfail(reason=reason),
    )


# Get all dtypes, except bool, float16 and complex
dtypes = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=False, no_complex=True
)
# Get all dtypes, except bool, float16, None and complex
dtypes_no_none = get_all_dtypes(
    no_bool=True, no_float16=True, no_none=True, no_complex=True
)
# Get all dtypes, except bool, float16, None, int (all) and complex
dtypes_float_only = get_all_dtypes(
    no_bool=True, no_float16=True, no_int=True, no_none=True, no_complex=True
)
usm_types = ["device", "shared", "host"]
ranges = [
    [1, None, None],  # 0
    [1.0, None, None],
    [1, 10, None],
    [1, 10, 1],
    [-10, -1, 1],
    [11, 41, 7],  # 5
    [1, 10, 1.0],
    [1, 10.0, 1],
    [0.7, 0.91, 0.03],
    [-1003.345, -987.44, 0.73],
    [1.15, 2.75, 0.05],  # 10
    [0.75, 10.23, 0.95],
    [10.23, 0.75, -0.95],
    [-0.1, 1.75, 0.1],
    get_xfail_test([0, 0, 1], "Can't allocate an empty array"),
    get_xfail_test([-1.0, 10, -2], "Impossible range"),  # 15
    get_xfail_test([-10, -1, -1], "Impossible range"),
    get_xfail_test([-1.0, None, None], "Can't allocate an empty array"),  # 17
]


@pytest.mark.parametrize("range", ranges)
@pytest.mark.parametrize("dtype", dtypes)
def test_dpnp_arange_default(range, dtype):
    """Tests `dpnp.arange()` overload with default setting.

    Test over all ranges and dtypes with default settings for `dpnp.arange()`.

    Args:
        range (list): A `list` of `start`, `stop` and `step` of the interval.
        dtype (type): A `type` to be used in the `dtype` parameter.

    Returns:
        None: Nothing.
    """
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


@pytest.mark.parametrize("range", ranges[0:-1])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_arange_from_device(range, dtype, usm_type):
    """Test device only `dpnp.arange()` overload with parameterized `usm_type`.

    We are skipping the last parameter since it's going to pass on that case.
    Because we are not comparing the results from original `dpnp.arange()` in
    this example.

    Args:
        range (list): A `list` of `start`, `stop` and `step` of the interval.
        dtype (type): A `type` to be used in the `dtype` parameter.
        usm_type (str): A `str` value to denote the type of USM one of
            `["device", "shared", "host"]`.

    Returns:
        None: Nothing.
    """
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


@pytest.mark.parametrize("range", ranges[0:-1])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
def test_dpnp_arange_from_queue(range, dtype, usm_type):
    """Test `dpnp.arange()` overload with specied queue and parameterized
        `usm_type`.

    We are skipping the last parameter since it's going to pass on that case.
    Because we are not comparing the results from original `dpnp.arange()` in
    this example.

    Args:
        range (list): A `list` of `start`, `stop` and `step` of the interval.
        dtype (type): A `type` to be used in the `dtype` parameter.
        usm_type (str): A `str` value to denote the type of USM one of
            `["device", "shared", "host"]`.

    Returns:
        None: Nothing.
    """
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


@pytest.mark.parametrize("range", ranges[3:8])
@pytest.mark.parametrize("start_dtype", dtypes_no_none)
@pytest.mark.parametrize("stop_dtype", dtypes_no_none)
@pytest.mark.parametrize("step_dtype", dtypes_no_none)
@pytest.mark.parametrize("dtype", dtypes)
def test_dpnp_arange_default_with_dtype_perm(
    range, start_dtype, stop_dtype, step_dtype, dtype
):
    """Tests dpnp.arange() overload with default setting and permutations of
        different `dtype`s for `start`, `stop` and `step`.

    From parameter case 3rd to 7th are used in this test. The parameters after
    8th case is not applicable since `step` < 1.0. They will fail on `int` types.
    NOTE: This unit-test is long since it's going to test 1600 cases.

    Args:
        range (list): A `list` of `start`, `stop` and `step` of the interval.
        start_dtype (type): The `dtype` for `start` value.
        stop_dtype (type): The `dtype` for `stop` value.
        step_dtype (type): The `dtype` for `step` value.
        dtype (type): A `type` to be used in the `dtype` parameter.

    Returns:
        None: Nothing.
    """
    start, stop, step = (
        start_dtype(range[0]),
        stop_dtype(range[1]),
        step_dtype(range[2]),
    )

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


@pytest.mark.parametrize("range", ranges[8:14])
@pytest.mark.parametrize("start_dtype", dtypes_float_only)
@pytest.mark.parametrize("stop_dtype", dtypes_float_only)
@pytest.mark.parametrize("step_dtype", dtypes_float_only)
@pytest.mark.parametrize("dtype", dtypes)
def test_dpnp_arange_default_with_float_only_dtype_perm(
    range, start_dtype, stop_dtype, step_dtype, dtype
):
    """Tests dpnp.arange() overload with default setting and permutations of
        different `dtype`s for `start`, `stop` and `step` with `float`s only.

    From parameter case 8th to 13th are used in this test. We will be testing
    on `float` types since `step` < 1.0.

    Args:
        range (list): A `list` of `start`, `stop` and `step` of the interval.
        start_dtype (type): The `dtype` for `start` value.
        stop_dtype (type): The `dtype` for `stop` value.
        step_dtype (type): The `dtype` for `step` value.
        dtype (type): A `type` to be used in the `dtype` parameter.

    Returns:
        None: Nothing.
    """
    start, stop, step = (
        start_dtype(range[0]),
        stop_dtype(range[1]),
        step_dtype(range[2]),
    )

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
