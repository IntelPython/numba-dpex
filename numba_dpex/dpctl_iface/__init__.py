# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The ``dpctl_iface`` module implements Numba's interface to the dpctl library
that provides Python and C bindings to DPC++'s SYCL runtime API. The module
includes:

- LLVM IR builders for dpctl C API functions to be called directly from a Numba
  generated LLVM module.
- Functions to launch kernels on the dpctl "current queue".

"""
import numba_dpex.dpctl_iface.dpctl_function_types as dpctl_fn_ty
from numba_dpex.dpctl_iface.dpctl_capi_fn_builder import DpctlCAPIFnBuilder
from numba_dpex.dpctl_iface.legacy_kernel_launch_ops import KernelLaunchOps

__all__ = [
    DpctlCAPIFnBuilder,
    KernelLaunchOps,
]

get_current_queue = dpctl_fn_ty.dpctl_get_current_queue()
malloc_shared = dpctl_fn_ty.dpctl_malloc_shared()
queue_memcpy = dpctl_fn_ty.dpctl_queue_memcpy()
free_with_queue = dpctl_fn_ty.dpctl_free_with_queue()
event_wait = dpctl_fn_ty.dpctl_event_wait()
event_delete = dpctl_fn_ty.dpctl_event_delete()
queue_wait = dpctl_fn_ty.dpctl_queue_wait()
