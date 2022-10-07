# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dpex.tests._helper import (
    dpnp_debug,
    filter_strings_with_skips_for_opencl,
    skip_no_dpnp,
)

from ._helper import wrapper_function

pytestmark = skip_no_dpnp

list_of_dtypes = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 100
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


list_of_shape = [
    (100),
    (50, 2),
    (10, 5, 2),
]


@pytest.fixture(params=list_of_shape)
def get_shape(request):
    return request.param


list_of_unary_ops = [
    "max",
    "amax",
    "min",
    "amin",
    "median",
    "mean",
    "cov",
]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return (
        wrapper_function("a", f"np.{request.param}(a)", globals()),
        request.param,
    )


@pytest.mark.parametrize("filter_str", filter_strings_with_skips_for_opencl)
def test_unary_ops(filter_str, unary_op, input_arrays, get_shape, capfd):
    a = input_arrays[0]
    op, name = unary_op
    if name != "cov":
        a = np.reshape(a, get_shape)

    actual = np.empty(shape=a.shape, dtype=a.dtype)
    expected = np.empty(shape=a.shape, dtype=a.dtype)

    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(a)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

    expected = op(a)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
