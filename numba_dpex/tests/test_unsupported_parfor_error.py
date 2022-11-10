# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numba
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestUnsupportedParforError:
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
        with pytest.raises(dpex.core.exceptions.UnsupportedParforError):
            device = dpctl.SyclDevice(device)
            with dpctl.device_context(device):
                fn = numba.njit(inner_call_fallback)
                fn()

    @pytest.mark.skip
    def test_fallback_reductions(self):
        def reduction(a):
            b = 1
            for i in numba.prange(len(a)):
                b += a[i]
            return b

        a = np.ones(10)
        device = dpctl.SyclDevice("opencl:gpu")
        with pytest.raises(dpex.core.exceptions.UnsupportedParforError):
            device = dpctl.SyclDevice(device)
            with dpctl.device_context(device):
                fn = numba.njit(reduction)
                fn(a)
