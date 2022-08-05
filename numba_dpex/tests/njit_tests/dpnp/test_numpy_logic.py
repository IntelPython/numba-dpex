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

from numba_dpex.tests._helper import dpnp_debug, filter_strings, skip_no_dpnp

pytestmark = skip_no_dpnp


@pytest.mark.parametrize("filter_str", filter_strings)
@pytest.mark.parametrize(
    "dtype", [np.bool_, np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize(
    "shape",
    [(0,), (4,), (2, 3)],
    ids=["(0,)", "(4,)", "(2, 3)"],
)
def test_all(dtype, shape, filter_str):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2**size):
        t = i

        a = np.empty(size, dtype=dtype)

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
