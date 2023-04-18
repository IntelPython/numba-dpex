# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for checking enforcing CFD in parfor pass.
"""


import dpctl
import dpnp
import numba as nb
import pytest

from numba_dpex import dpjit
from numba_dpex.core.exceptions import ComputeFollowsDataInferenceError
from numba_dpex.tests._helper import skip_no_opencl_gpu

shapes = [10, (2, 5)]
dtypes = [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64]
usm_types = ["device"]
devices = ["gpu"]


@dpjit
def func1(a, b):
    c = a + b
    return c


@skip_no_opencl_gpu
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("usm_type", usm_types)
@pytest.mark.parametrize("device", devices)
def test_parfor_legalize_cfd_pass(shape, dtype, usm_type, device):
    a = dpnp.zeros(shape=shape, dtype=dtype, usm_type=usm_type, device=device)
    b = dpnp.ones(shape=shape, dtype=dtype, usm_type=usm_type, device=device)
    try:
        c = func1(a, b)
    except Exception:
        pytest.fail("Running Parfor CFD Pass check failed")

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


@skip_no_opencl_gpu
def test_parfor_legalize_cfd_pass_raise():
    a = dpnp.zeros(shape=10, device="cpu")
    b = dpnp.ones(shape=10, device="gpu")

    with pytest.raises(ComputeFollowsDataInferenceError):
        func1(a, b)


@skip_no_opencl_gpu
def test_cfd_error_due_to_lhs():
    a = dpnp.zeros(shape=10, device="cpu")
    b = dpnp.ones(shape=10, device="cpu")

    @dpjit
    def vecadd_prange(a, b):
        c = dpnp.empty(a.shape, dtype=a.dtype, device="gpu")
        for idx in nb.prange(a.size):
            c[idx] = a[idx] + b[idx]
        return c

    with pytest.raises(ComputeFollowsDataInferenceError):
        vecadd_prange(a, b)
