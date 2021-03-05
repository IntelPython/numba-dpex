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

import pytest
import dpctl
from numba_dppy.context import device_context
from numba_dppy.tests.skip_tests import skip_test


list_of_filter_strs = [
    "opencl:gpu:0",
    "level0:gpu:0",
    "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


def test_dpctl_api(filter_str):
    if skip_test(filter_str):
        pytest.skip()

    with device_context(filter_str) as gpu_queue:
        dpctl.dump()
        dpctl.get_current_queue()
        dpctl.get_num_platforms()
        dpctl.get_num_activated_queues()
        dpctl.has_cpu_queues()
        dpctl.has_gpu_queues()
        dpctl.has_sycl_platforms()
        dpctl.is_in_device_context()
