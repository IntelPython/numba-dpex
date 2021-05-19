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
from numba_dppy.tests._helper import skip_test
import dpctl.memory as dpctl_mem


def test_array_as_arg_context_mismatch():
    @dppy.kernel
    def sample_kernel(a):
        i = dppy.get_global_id(0)
        a[i] = a[i] + 1

    a = np.array(np.random.random(1023), np.int32)

    with dpctl.device_context("opencl:gpu:0") as gpu_queue:
        # Users can explicitly provide SYCL queue, the queue used internally
        # will be the current queue. Current queue is set by the context_manager
        # dpctl.device_context.
        inp_buf = dpctl_mem.MemoryUSMShared(a.size * a.dtype.itemsize, queue=gpu_queue)
        inp_ndarray = np.ndarray(a.shape, buffer=inp_buf, dtype=a.dtype)
        np.copyto(inp_ndarray, a)

    with dpctl.device_context("level_zero:gpu:0") as gpu_queue:
        with pytest.raises(Exception) as e_info:
            sample_kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](inp_ndarray)

    with dpctl.device_context("opencl:gpu:0") as gpu_queue:
        sample_kernel[a.size, dppy.DEFAULT_LOCAL_SIZE](inp_ndarray)

    assert np.all(a + 1 == inp_ndarray)
