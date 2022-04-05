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
Tests for boxing for dpnp.ndarray
"""

import pytest
from dpctl.tensor.numpy_usm_shared import ndarray as dpctl_ndarray
from dpnp import ndarray as dpnp_ndarray
from numba import njit

dpnp_mark = pytest.mark.xfail(raises=TypeError, reason="No unboxing")


@pytest.mark.parametrize(
    "array",
    [
        dpctl_ndarray([1]),
        pytest.param(dpnp_ndarray([1]), marks=dpnp_mark),
    ],
)
def test_njit(array):
    @njit
    def func(a):
        return a

    func(array)
