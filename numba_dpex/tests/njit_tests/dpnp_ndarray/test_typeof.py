# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for numba_dpex.dpnp_ndarray.typeof
"""

import pytest
from dpctl.tensor.numpy_usm_shared import ndarray as dpctl_ndarray
from dpnp import ndarray as dpnp_ndarray
from numba import njit, typeof, types

from numba_dpex.dpnp_ndarray import dpnp_ndarray_Type
from numba_dpex.numpy_usm_shared import UsmSharedArrayType


@pytest.mark.parametrize(
    "array_type, expected_numba_type",
    [
        (dpctl_ndarray, UsmSharedArrayType),
        (dpnp_ndarray, dpnp_ndarray_Type),
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
    expected_type = expected_numba_type(types.float64, expected_ndim, "C")
    assert typeof(array) == expected_type
