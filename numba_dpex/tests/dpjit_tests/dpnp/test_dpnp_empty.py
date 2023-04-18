# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for dpnp ndarray constructors."""

import dpctl
import dpnp
import pytest

from numba_dpex import dpjit

shapes = [11, (2, 5)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device", "shared", "host"]
devices = ["cpu", None]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_dpnp_empty(shape, dtype, usm_type, device):
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
    if device is not None:
        assert (
            c.sycl_device.filter_string
            == dpctl.SyclDevice(device).filter_string
        )
    else:
        c.sycl_device.filter_string == dpctl.SyclDevice().filter_string


@pytest.mark.parametrize("shape", shapes)
def test_dpnp_empty_default_dtype(shape):
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
