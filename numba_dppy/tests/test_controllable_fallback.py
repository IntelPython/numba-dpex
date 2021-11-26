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
import warnings

import dpctl
import numba
import numpy as np

from numba_dppy import config

from . import _helper


@pytest.mark.skipif(not _helper.has_gpu_queues(), reason="test only on GPU system")
class TestDPPYFallback:
    def test_dppy_fallback_true(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        config.DEBUG = 1
        with warnings.catch_warnings(record=True) as w:
            device = dpctl.SyclDevice("opencl:gpu")
            with dpctl.device_context(device):
                dppy = numba.njit(parallel=True)(inner_call_fallback)
                dppy_fallback_true = dppy()

        ref_result = inner_call_fallback()
        config.DEBUG = 0

        np.testing.assert_array_equal(dppy_fallback_true, ref_result)
        assert "Failed to offload parfor" in str(w[-1].message)

    @pytest.mark.xfail
    def test_dppy_fallback_false(self):
        @numba.jit
        def fill_value(i):
            return i

        def inner_call_fallback():
            x = 10
            a = np.empty(shape=x, dtype=np.float32)

            for i in numba.prange(x):
                a[i] = fill_value(i)

            return a

        try:
            config.DEBUG = 1
            config.FALLBACK_ON_CPU = 0
            with warnings.catch_warnings(record=True) as w:
                device = dpctl.SyclDevice("opencl:gpu")
                with dpctl.device_context(device):
                    dppy = numba.njit(parallel=True)(inner_call_fallback)
                    dppy_fallback_false = dppy()

        finally:
            ref_result = inner_call_fallback()
            config.FALLBACK_ON_CPU = 1
            config.DEBUG = 0

            not np.testing.assert_array_equal(dppy_fallback_false, ref_result)
            assert not "Failed to offload parfor" in str(w[-1].message)
