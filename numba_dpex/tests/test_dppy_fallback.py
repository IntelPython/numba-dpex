# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dpctl
import numba
import numpy as np

from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestFallback:
    def test_fallback_inner_call(self):
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
            fn = numba.njit(inner_call_fallback)
            result = fn()

        ref_result = inner_call_fallback()

        np.testing.assert_array_equal(result, ref_result)
        assert "Failed to offload parfor " in str(w[-1].message)

    def test_fallback_reductions(self):
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
            fn = numba.njit(reduction)
            result = fn(a)

        ref_result = reduction(a)

        np.testing.assert_array_equal(result, ref_result)
        assert "Failed to offload parfor " in str(w[-1].message)
