#! /usr/bin/env python
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

import contextlib
import dpctl
from numba_dppy import config


def has_gpu_queues(backend="opencl"):
    """
    Checks if dpctl is able to select a GPU device that defaults to
    an OpenCL GPU.
    """
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))


def has_cpu_queues(backend="opencl"):
    """
    Checks if dpctl is able to select a CPU device that defaults to
    an OpenCL CPU.
    """
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))


def has_sycl_platforms():
    """
    Checks if dpctl is able to identify a non-host SYCL platform.
    """
    platforms = dpctl.get_platforms()
    for p in platforms:
        if not p.backend is dpctl.backend_type.host:
            return True
    return False


def is_gen12(device_type):
    with dpctl.device_context(device_type):
        q = dpctl.get_current_queue()
        device = q.get_sycl_device()
        name = device.name
        if "Gen12" in name:
            return True

        return False


def platform_not_supported(device_type):
    import platform

    platform = platform.system()
    device = device_type.split(":")[0]

    if device == "level_zero" and platform == "Windows":
        return True

    return False


def skip_test(device_type):
    skip = False
    try:
        with dpctl.device_context(device_type):
            pass
    except Exception:
        skip = True

    if not skip:
        if platform_not_supported(device_type):
            skip = True

    return skip


@contextlib.contextmanager
def override_config(name, value, config=config):
    """
    Extends `numba/tests/support.py:override_config()` with argument `config`
    which is `numba_dppy.config` by default.
    """
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)
