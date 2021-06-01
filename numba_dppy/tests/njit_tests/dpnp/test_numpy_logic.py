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
from numba import njit
import numba_dppy
import pytest
from numba_dppy.testing import dpnp_debug
from .dpnp_skip_test import dpnp_skip_test as skip_test


list_of_filter_strs = [
    "opencl:gpu:0",
    "level_zero:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


@pytest.mark.parametrize("shape",
                         [(0,), (4,), (2, 3), (2, 2, 2)],
                         ids=['(0,)', '(4,)', '(2,3)', '(2,2,2)'])
def test_all(shape, filter_str):
    if skip_test(filter_str):
        pytest.skip()

    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2 ** size):
        t = i

        a = np.empty(size, dtype=np.bool)

        for j in range(size):
            a[j] = 0 if t % 2 == 0 else j + 1
            t = t >> 1

        a = a.reshape(shape)

        def fn(a):
            return np.all(a)

        f = njit(fn)
        with dpctl.device_context(filter_str), dpnp_debug():
            actual = f(a)

        expected = fn(a)
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
