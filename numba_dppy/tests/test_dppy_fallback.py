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

import warnings

import dpctl
import numba
import numpy as np
from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestDPPYFallback:
    def test_dppy_fallback_inner_call(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        device = dpctl.SyclDevice("opencl:gpu")
        with warnings.catch_warnings(record=True) as w, dpctl.device_context(
            device
        ):
            dppy = numba.njit(inner_call_fallback)
            dppy_result = dppy()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(dppy_result, ref_result)
        assert "Failed to offload parfor " in str(w[-1].message)

    def test_dppy_fallback_reductions(self):
        def reduction(a):
            b = 1
            for i in numba.prange(len(a)):
                b += a[i]
            return b

        a = np.ones(10)
        device = dpctl.SyclDevice("opencl:gpu")
        with warnings.catch_warnings(record=True) as w, dpctl.device_context(
            device
        ):
            dppy = numba.njit(reduction)
            dppy_result = dppy(a)

        ref_result = reduction(a)

        np.testing.assert_array_equal(dppy_result, ref_result)
        assert "Failed to offload parfor " in str(w[-1].message)
