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

import numba_dppy as dpex
from numba_dppy.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_kernel_using_same_queue(filter_str):
    """Test kernel caching when the same queue is used to submit a kernel
    multiple times.

    Args:
        filter_str: SYCL filter selector string
    """
    global_size = 10
    N = global_size

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    with dpctl.device_context(filter_str) as gpu_queue:
        func = dpex.kernel(data_parallel_sum)
        cached_kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
            func._get_argtypes(a, b, c), gpu_queue
        )

        for i in range(10):
            _kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
                func._get_argtypes(a, b, c), gpu_queue
            )
            assert _kernel == cached_kernel


@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_kernel_using_same_context(filter_str):
    """Test kernel caching for the scenario where different SYCL queues that
    share a SYCL context are used to submit a kernel.

    Args:
        filter_str: SYCL filter selector string
    """
    global_size = 10
    N = global_size

    def data_parallel_sum(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    # Set the global queue to the default device so that the cached_kernel gets
    # created for that device
    dpctl.set_global_queue(filter_str)
    func = dpex.kernel(data_parallel_sum)
    default_queue = dpctl.get_current_queue()
    cached_kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
        func._get_argtypes(a, b, c), default_queue
    )
    for i in range(0, 10):
        # Each iteration create a fresh queue that will share the same context
        with dpctl.device_context(filter_str) as gpu_queue:
            _kernel = func[global_size, dpex.DEFAULT_LOCAL_SIZE].specialize(
                func._get_argtypes(a, b, c), gpu_queue
            )
            assert _kernel == cached_kernel
