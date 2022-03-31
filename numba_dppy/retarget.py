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

try:
    from numba.core.dispatcher import TargetConfigurationStack
except ImportError:
    # for support numba 0.54 and <=0.55.0dev0=*_469
    from numba.core.dispatcher import TargetConfig as TargetConfigurationStack

from numba.core.retarget import BasicRetarget

from numba_dppy.target import DPEX_TARGET_NAME


class DpexRetarget(BasicRetarget):
    def __init__(self, filter_str):
        self.filter_str = filter_str
        super(DpexRetarget, self).__init__()

    @property
    def output_target(self):
        return DPEX_TARGET_NAME

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=DPEX_TARGET_NAME)(cpu_disp.py_func)
        return kernel


_first_level_cache = dict()


def _retarget(sycl_queue):
    filter_string = sycl_queue.sycl_device.filter_string

    result = _first_level_cache.get(filter_string)

    if not result:
        result = DpexRetarget(filter_string)
        _first_level_cache[filter_string] = result

    return result


def _retarget_context_manager(sycl_queue):
    """Return context manager for retargeting njit offloading."""
    retarget = _retarget(sycl_queue)
    return TargetConfigurationStack.switch_target(retarget)


def _register_context_factory():
    dpctl.nested_context_factories.append(_retarget_context_manager)


_register_context_factory()
offload_to_sycl_device = dpctl.device_context
