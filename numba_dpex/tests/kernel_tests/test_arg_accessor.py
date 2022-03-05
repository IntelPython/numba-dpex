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

import dpctl
import numpy as np
import pytest

import numba_dpex as dppy
from numba_dpex.tests._helper import filter_strings


def call_kernel(global_size, local_size, A, B, C, func):
    func[global_size, local_size](A, B, C)


global_size = 10
local_size = 1
N = global_size * local_size


def sum_kernel(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


list_of_dtypes = [
    np.float32,
    np.float64,
]


@pytest.fixture(params=list_of_dtypes)
def input_arrays(request):
    # The size of input and out arrays to be used
    a = np.array(np.random.random(N), request.param)
    b = np.array(np.random.random(N), request.param)
    c = np.zeros_like(a)
    return a, b, c


list_of_kernel_opt = [
    {"read_only": ["a", "b"], "write_only": ["c"], "read_write": []},
    {},
]


@pytest.fixture(params=list_of_kernel_opt)
def kernel(request):
    return dppy.kernel(access_types=request.param)(sum_kernel)


@pytest.mark.parametrize("filter_str", filter_strings)
def test_kernel_arg_accessor(filter_str, input_arrays, kernel):
    a, b, actual = input_arrays
    expected = a + b
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        call_kernel(global_size, local_size, a, b, actual, kernel)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0)
