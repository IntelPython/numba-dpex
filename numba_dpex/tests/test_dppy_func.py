# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np

import numba_dpex as dpex
from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestFunc:
    N = 257

    def test_func_device_array(self):
        @dpex.func
        def g(a):
            return a + 1

        @dpex.kernel
        def f(a, b):
            i = dpex.get_global_id(0)
            b[i] = g(a[i])

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, dpex.DEFAULT_LOCAL_SIZE](a, b)

        assert np.all(b == 2)

    def test_func_ndarray(self):
        @dpex.func
        def g(a):
            return a + 1

        @dpex.kernel
        def f(a, b):
            i = dpex.get_global_id(0)
            b[i] = g(a[i])

        @dpex.kernel
        def h(a, b):
            i = dpex.get_global_id(0)
            b[i] = g(a[i]) + 1

        a = np.ones(self.N)
        b = np.ones(self.N)

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            f[self.N, dpex.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 2)

            h[self.N, dpex.DEFAULT_LOCAL_SIZE](a, b)

            assert np.all(b == 3)
