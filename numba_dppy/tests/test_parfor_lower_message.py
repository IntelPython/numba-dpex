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

import dpctl
import numpy as np
from numba import njit, prange
from numba.tests.support import captured_stdout

import numba_dppy as dppy
from numba_dppy import config

from . import _helper


def prange_example():
    n = 10
    a = np.ones((n), dtype=np.float64)
    b = np.ones((n), dtype=np.float64)
    c = np.ones((n), dtype=np.float64)
    for i in prange(n // 2):
        a[i] = b[i] + c[i]

    return a


@pytest.mark.skipif(not _helper.has_gpu_queues(), reason="test only on GPU system")
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
