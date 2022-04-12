# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
