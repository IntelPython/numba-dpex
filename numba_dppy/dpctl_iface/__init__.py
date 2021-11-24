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
The ``dpctl_iface`` module implements Numba's interface to the dpctl library
that provides Python and C bindings to DPC++'s SYCL runtime API. The module
includes:

- LLVM IR builders for dpctl C API functions to be called directly from a Numba
  generated LLVM module.
- Functions to lauch kernels on the dpctl "current queue".

"""
import numba_dppy.dpctl_iface.dpctl_function_types as dpctl_fn_ty
from numba_dppy.dpctl_iface.dpctl_capi_fn_builder import DpctlCAPIFnBuilder
from numba_dppy.dpctl_iface.kernel_launch_ops import KernelLaunchOps
from numba_dppy.dpctl_iface.usm_ndarray_type import USMNdArrayType

__all__ = [
    DpctlCAPIFnBuilder,
    KernelLaunchOps,
    USMNdArrayType,
]

get_current_queue = dpctl_fn_ty.dpctl_get_current_queue()
malloc_shared = dpctl_fn_ty.dpctl_malloc_shared()
queue_memcpy = dpctl_fn_ty.dpctl_queue_memcpy()
free_with_queue = dpctl_fn_ty.dpctl_free_with_queue()
event_wait = dpctl_fn_ty.dpctl_event_wait()
event_delete = dpctl_fn_ty.dpctl_event_delete()
queue_wait = dpctl_fn_ty.dpctl_queue_wait()
