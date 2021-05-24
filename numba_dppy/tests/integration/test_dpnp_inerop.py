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
import dpctl.tensor as dpt
import numpy as np
from numba import njit
import pytest
import numba_dppy as dppy
import dpnp

list_of_dtype = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]

@pytest.fixture(params=list_of_dtype)
def dtype(request):
    return request.param


list_of_usm_type = [
    "shared",
    "device",
    #"host",
]


@pytest.fixture(params=list_of_usm_type)
def usm_type(request):
    return request.param


def test_consuming_array_from_dpnp(offload_device, dtype, usm_type):
    '''
    @dppy.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dppy.get_global_id(0)
        c[i] = a[i] + b[i]

    global_size = 1021

    b = np.array(np.random.random(global_size), dtype=np.float32)
    c = np.ones_like(b)
    '''
    global_size = 1021
    with dpctl.device_context("opencl:gpu"):
        a = dpnp.arange(global_size, dtype=dtype)

        #db = dpt.usm_ndarray(b.shape, dtype=b.dtype, buffer="shared")
        #db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

        #data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, db, c)
