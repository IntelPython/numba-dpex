# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import dpctl
import numba_dpex as dppy
import numpy as np
import pytest
from numba_dpex.tests._helper import filter_strings

list_of_unary_ops = ["fabs", "exp", "log", "sqrt", "sin", "cos", "tan"]


@pytest.fixture(params=list_of_unary_ops)
def unary_op(request):
    return request.param


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    N = 2048
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    return a, b


@pytest.mark.parametrize("filter_str", filter_strings)
def test_binary_ops(filter_str, unary_op, input_arrays):
    a, actual = input_arrays
    uop = getattr(math, unary_op)
    np_uop = getattr(np, unary_op)

    @dppy.kernel
    def f(a, b):
        i = dppy.get_global_id(0)
        b[i] = uop(a[i])

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        f[a.size, dppy.DEFAULT_LOCAL_SIZE](a, actual)

    expected = np_uop(a)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
