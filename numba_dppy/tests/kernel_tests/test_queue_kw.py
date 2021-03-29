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
from numba_dppy.tests.skip_tests import skip_test


list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


def dppy_kernel(a, b):
    idx = dppy.get_global_id(0)
    b[idx] = a[idx] + 1


def test_queue_kw(filter_str):
    kernel = dppy.kernel(queue=filter_str)(dppy_kernel)

    with dpctl.device_context(filter_str) as gpu_queue:
        assert kernel.sycl_queue.equals(gpu_queue)
