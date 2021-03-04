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

import dpctl
from numba_dppy.context import device_context


def is_gen12(device_type):
    with device_context(device_type):
        q = dpctl.get_current_queue()
        device = q.get_sycl_device()
        name = device.get_device_name()
        if "Gen12" in name:
            return True

        return False


def platform_not_supported(device_type):
    import platform

    platform = platform.system()
    device = device_type.split(":")[0]

    if device == "level0" and platform == "Windows":
        return True

    return False


def skip_test(device_type):
    skip = False
    try:
        with device_context(device_type):
            pass
    except Exception:
        skip = True

    if not skip:
        if platform_not_supported(device_type):
            skip = True

    return skip
