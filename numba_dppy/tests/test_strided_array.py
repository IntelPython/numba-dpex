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

import numpy as np
import os
import pytest
import dpctl

import numba_dppy
from numba_dppy import config
from numba_dppy.tests._helper import skip_test


@numba_dppy.kernel
def sum(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def test_strided_array_kernel(offload_device):
    if skip_test(offload_device):
        pytest.skip()

    global_size = 606
    a = np.arange(global_size * 2, dtype="i4")[::2]
    b = np.arange(global_size, dtype="i4")[::-1]
    got = np.zeros(global_size, dtype="i4")

    expected = a + b

    with dpctl.device_context(offload_device):
        sum[global_size, numba_dppy.DEFAULT_LOCAL_SIZE](a, b, got)

    assert np.array_equal(expected, got)
