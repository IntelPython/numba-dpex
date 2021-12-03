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

import numba_dppy as dppy

from . import _helper


@pytest.mark.skipif(
    not _helper.has_opencl_gpu(), reason="test only on GPU system"
)
class TestDPPYFunc:
    N = 257

    def test_dppy_func_device_array(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

        assert np.all(b == 2)

    def test_dppy_func_ndarray(self):
        @dppy.func
        def g(a):
            return a + 1

        @dppy.kernel
        def f(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i])

        @dppy.kernel
        def h(a, b):
            i = dppy.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 2)

            h[self.N, dppy.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 3)
