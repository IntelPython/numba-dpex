# Copyright 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
from numba import njit, prange
from numba.tests.support import captured_stdout

import numba_dpex as dpex
from numba_dpex import config as dpex_config
from numba_dpex.tests._helper import skip_no_opencl_gpu


@skip_no_opencl_gpu
class TestOffloadDiagnostics:
    def test_parfor(self):
        def prange_func():
            n = 10
            a = np.ones((n), dtype=np.float64)
            b = np.ones((n), dtype=np.float64)
            c = np.ones((n), dtype=np.float64)
            for i in prange(n // 2):
                a[i] = b[i] + c[i]

            return a

        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            dpex_config.OFFLOAD_DIAGNOSTICS = 1
            jitted = njit(parallel=True)(prange_func)

            with captured_stdout() as got:
                jitted()

            dpex_config.OFFLOAD_DIAGNOSTICS = 0
            assert "Auto-offloading" in got.getvalue()
            assert "Device -" in got.getvalue()
