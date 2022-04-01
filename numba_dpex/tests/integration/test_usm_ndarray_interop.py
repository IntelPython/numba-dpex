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
import pytest
from numba import njit

import numba_dpex as dpex
from numba_dpex.tests._helper import filter_strings

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
    "host",
]


@pytest.fixture(params=list_of_usm_type)
def usm_type(request):
    return request.param


@pytest.mark.parametrize("filter_str", filter_strings)
def test_consuming_usm_ndarray(filter_str, dtype, usm_type):
    @dpex.kernel
    def data_parallel_sum(a, b, c):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        j = dpex.get_global_id(1)
        c[i, j] = a[i, j] + b[i, j]

    N = 1021
    global_size = N * N

    a = np.array(np.random.random(global_size), dtype=dtype).reshape(N, N)
    b = np.array(np.random.random(global_size), dtype=dtype).reshape(N, N)

    got = np.ones_like(a)

    with dpctl.device_context(filter_str) as gpu_queue:
        da = dpt.usm_ndarray(
            a.shape,
            dtype=a.dtype,
            buffer=usm_type,
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        da.usm_data.copy_from_host(a.reshape((-1)).view("|u1"))

        db = dpt.usm_ndarray(
            b.shape,
            dtype=b.dtype,
            buffer=usm_type,
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        db.usm_data.copy_from_host(b.reshape((-1)).view("|u1"))

        dc = dpt.usm_ndarray(
            got.shape,
            dtype=got.dtype,
            buffer=usm_type,
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

        data_parallel_sum[(N, N), dpex.DEFAULT_LOCAL_SIZE](da, db, dc)

        dc.usm_data.copy_to_host(got.reshape((-1)).view("|u1"))

        expected = a + b

        assert np.array_equal(got, expected)
