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
from numba._dispatcher import set_use_tls_target_stack
from numba.core.dispatcher import TargetConfig

from numba_dppy.dppy_offload_dispatcher import DppyOffloadDispatcher


@contextmanager
def switch_target(retarget):
    # __enter__
    tc = TargetConfig()
    tc.push(retarget)
    set_use_tls_target_stack(True)
    yield
    # __exit__
    tc.pop()
    set_use_tls_target_stack(False)


def retarget_to_gpu(cpu_disp):
    dispatcher = DppyOffloadDispatcher(cpu_disp.py_func)
    return lambda *args, **kwargs: dispatcher(*args, **kwargs)


@contextmanager
def device_context(*args, **kwargs):
    with switch_target(retarget_to_gpu):
        with dpctl.device_context(*args, **kwargs) as queue:
            yield queue
