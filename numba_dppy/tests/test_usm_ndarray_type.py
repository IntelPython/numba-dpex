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
import numba_dppy as dppy
import pytest
import dpctl
import dpctl.tensor as dpt
from numba.misc.special import typeof
from numba_dppy.tests._helper import skip_test
from numba_dppy.driver import USM_NdArrayType

list_of_dtypes = [
    np.int32,
    np.float32,
    np.int64,
    np.float64,
]

@pytest.fixture(params=list_of_dtypes)
def input_array(request):
    a = np.array(np.random.random(10), request.param)
    da = dpt.usm_ndarray(a.shape, dtype=a.dtype, buffer="shared")

    return da


def test_usm_ndarray_type(offload_device, input_array):
    if skip_test(offload_device):
        pytest.skip()

    assert isinstance(typeof(input_array), USM_NdArrayType)
