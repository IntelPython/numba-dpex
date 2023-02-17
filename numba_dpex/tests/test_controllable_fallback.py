# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dpctl
import numba
import numpy as np
import pytest

from numba_dpex import config
from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestParforFallback:
    def test_parfor_fallback_true(self):
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
                fn = numba.njit(parallel=True)(inner_call_fallback)
                fallback_true = fn()

        ref_result = inner_call_fallback()
        config.DEBUG = 0

        np.testing.assert_array_equal(fallback_true, ref_result)
        assert "Failed to offload parfor" in str(w[-1].message)

    @pytest.mark.xfail
    def test_parfor_fallback_false(self):
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
                    fn = numba.njit(parallel=True)(inner_call_fallback)
                    fallback_false = fn()

        finally:
            ref_result = inner_call_fallback()
            config.FALLBACK_ON_CPU = 1
            config.DEBUG = 0

            not np.testing.assert_array_equal(fallback_false, ref_result)
            assert "Failed to offload parfor" not in str(w[-1].message)
