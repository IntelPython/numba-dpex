# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for numba_dpex.dpnp_ndarray.typeof
"""

import pytest
from dpnp import ndarray as dpnp_ndarray
from numba import typeof

from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray


@pytest.mark.parametrize(
    "array_type, expected_numba_type",
    [
        (dpnp_ndarray, DpnpNdArray),
    ],
)
@pytest.mark.parametrize(
    "shape, expected_ndim",
    [
        ([1], 1),
        ([1, 1], 2),
    ],
)
def test_typeof(array_type, shape, expected_numba_type, expected_ndim):
    array = array_type(shape)
    assert isinstance(typeof(array), expected_numba_type)
