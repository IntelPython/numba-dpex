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

from contextlib import contextmanager

import dpctl
from numba import njit
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget

from numba_dppy.target import DPPY_TARGET_NAME


class DPPYRetarget(BasicRetarget):
    def __init__(self, filter_str):
        self.filter_str = filter_str
        super(DPPYRetarget, self).__init__()

    @property
    def output_target(self):
        return DPPY_TARGET_NAME

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=DPPY_TARGET_NAME)(cpu_disp.py_func)
        return kernel


first_level_cache = dict()


@contextmanager
def offload_to_sycl_device(dpctl_device):
    with dpctl.device_context(dpctl_device) as sycl_queue:
        filter_string = sycl_queue.sycl_device.filter_string
        retarget = first_level_cache.get(filter_string, None)

        if retarget is None:
            retarget = DPPYRetarget(filter_string)
            first_level_cache[filter_string] = retarget
        with TargetConfigurationStack.switch_target(retarget):
            yield sycl_queue
