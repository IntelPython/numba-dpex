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

"""
The numba-dpex extension module adds data-parallel offload support to Numba.
"""
import numba.testing

from numba_dppy.interop import asarray
from numba_dppy.retarget import offload_to_sycl_device

from . import config

if config.HAS_NON_HOST_DEVICE:
    from .device_init import *
else:
    raise ImportError("No non-host SYCL device found to execute kernels.")


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = ["offload_to_sycl_device"]
