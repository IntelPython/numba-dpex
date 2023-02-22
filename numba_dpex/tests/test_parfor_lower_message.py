# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import numpy as np
from numba import njit, prange
from numba.tests.support import captured_stdout

from numba_dpex import config
from numba_dpex.tests._helper import skip_no_opencl_gpu


def prange_example():
    n = 10
    a = np.ones((n), dtype=np.float64)
    b = np.ones((n), dtype=np.float64)
    c = np.ones((n), dtype=np.float64)
    for i in prange(n // 2):
        a[i] = b[i] + c[i]

    return a


@skip_no_opencl_gpu
class TestParforMessage:
    def test_parfor_message(self):
        device = dpctl.SyclDevice("opencl:gpu")
        with dpctl.device_context(device):
            config.DEBUG = 1
            jitted = njit(prange_example)

            with captured_stdout() as got:
                jitted()

            config.DEBUG = 0
            assert "Parfor offloaded " in got.getvalue()
