################################################################################
#                                 Numba-DPPY
#
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
################################################################################

import dpctl
import numpy as np
import pytest
from numba import njit

import numba_dppy as dppy
from numba_dppy.tests._helper import dpnp_debug, filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize(
    "arr",
    [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    ids=["[1, 2, 3, 4]", "[1, 2, 3, 4, 5, 6, 7, 8, 9]"],
)
def test_repeat(filter_str, arr):
    a = np.array(arr)
    repeats = 2

    def fn(a, repeats):
        return np.repeat(a, repeats)

    f = njit(fn)
    with dpctl.device_context(filter_str), dpnp_debug():
        actual = f(a, repeats)

    expected = fn(a, repeats)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
