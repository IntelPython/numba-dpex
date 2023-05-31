# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import math

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy
import pytest

from numba_dpex import dpjit

shapes = [11, (3, 7)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]
devices = ["cpu", "unknown"]
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
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_dpnp_full(shape, fill_value, dtype, usm_type, device):
    @dpjit
    def func(shape, fill_value):
        c = dpnp.full(
            shape, fill_value, dtype=dtype, usm_type=usm_type, device=device
        )
        return c

    a = numpy.full(shape, fill_value, dtype=dtype)

    try:
        c = func(shape, fill_value)
    except Exception:
        pytest.fail("Calling dpnp.full inside dpjit failed")

    if len(c.shape) == 1:
        assert c.shape[0] == shape
    else:
        assert c.shape == shape

    assert c.dtype == dtype
    assert c.usm_type == usm_type
    if device != "unknown":
        assert (
            c.sycl_device.filter_string
            == dpctl.SyclDevice(device).filter_string
        )
    else:
        c.sycl_device.filter_string == dpctl.SyclDevice().filter_string

    assert numpy.array_equal(dpt.asnumpy(c._array_obj), a)
