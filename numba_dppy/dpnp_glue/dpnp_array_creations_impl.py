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

import numpy as np
from numba import types
from numba.core.extending import overload, register_jitable
from numba.core.typing import signature

import numba_dppy
import numba_dppy.dpnp_glue as dpnp_lowering
import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba_dppy import dpctl_functions

from . import stubs


@register_jitable
def common_impl(a, b, out, dpnp_func, PRINT_DEBUG):
    if a.size == 0:
        raise ValueError("Passed Empty array")

    sycl_queue = dpctl_functions.get_current_queue()

    b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, b_usm, b.ctypes, b.size * b.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    dpnp_func(out_usm, b_usm, a.size)

    event = dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    dpctl_functions.free_with_queue(b_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, out.size])

    if PRINT_DEBUG:
        print("dpnp implementation")


@register_jitable
def common_shape_impl(a, out, dpnp_func, PRINT_DEBUG):
    if a.size == 0:
        raise ValueError("Passed Empty array")

    sycl_queue = dpctl_functions.get_current_queue()

    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim)

    event = dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, out.size])

    if PRINT_DEBUG:
        print("dpnp implementation")


@overload(stubs.dpnp.zeros_like)
def dpnp_zeros_like_impl(a, dtype=None):
    name = "zeros_like"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_common.cpp#L224

    Function declaration:
    void dpnp_initval_c(void* result1, void* value, size_t size)

    """
    res_dtype = dtype
    if dtype == types.none or dtype is None:
        res_dtype = a.dtype
        name_dtype = res_dtype.name
    else:
        name_dtype = res_dtype.dtype.name

    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [name_dtype, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, dtype=None):
        b = np.zeros(1, dtype=res_dtype)
        out = np.zeros(a.shape, dtype=res_dtype)
        common_impl(a, b, out, dpnp_func, PRINT_DEBUG)
        return out

    return dpnp_impl


@overload(stubs.dpnp.ones_like)
def dpnp_ones_like_impl(a, dtype=None):
    name = "ones_like"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_common.cpp#L224

    Function declaration:
    void dpnp_initval_c(void* result1, void* value, size_t size)

    """
    res_dtype = dtype
    if dtype == types.none or dtype is None:
        res_dtype = a.dtype
        name_dtype = res_dtype.name
    else:
        name_dtype = res_dtype.dtype.name

    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [name_dtype, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, dtype=None):
        b = np.ones(1, dtype=res_dtype)
        out = np.ones(a.shape, dtype=res_dtype)
        common_impl(a, b, out, dpnp_func, PRINT_DEBUG)
        return out

    return dpnp_impl


@overload(stubs.dpnp.full_like)
def dpnp_full_like_impl(a, b):
    name = "full_like"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_common.cpp#L224

    Function declaration:
    void dpnp_initval_c(void* result1, void* value, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [b.dtype.name, "NONE"], sig)

    res_dtype = b.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, b):
        out = np.ones(a.shape, dtype=res_dtype)
        common_impl(a, b, out, dpnp_func, PRINT_DEBUG)
        return out

    return dpnp_impl


# TODO: This implementation is incorrect
@overload(stubs.dpnp.full)
def dpnp_full_impl(a, b):
    name = "full"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_arraycreation.cpp#L70

    Function declaration:
    void dpnp_full_c(void* array_in, void* result, const size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [b.dtype.name, "NONE"], sig)

    res_dtype = b.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, b):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, b_usm, b.ctypes, b.size * b.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out = np.arange(0, a.size, 1, res_dtype)
        out_usm = dpctl_functions.malloc_shared(
            out.size * out.itemsize, sycl_queue
        )

        dpnp_func(b_usm, out_usm, a.size)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(b_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return out

    return dpnp_impl


@overload(stubs.dpnp.trace)
def dpnp_trace_impl(a):
    name = "trace"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.6.2/dpnp/backend/kernels/dpnp_krnl_arraycreation.cpp#L218

    Function declaration:
    void dpnp_trace_c(const void* array1_in, void* result1, const size_t* shape_, const size_t ndim)

    """
    sig = signature(
        ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        diag_arr = numba_dppy.dpnp.diagonal(a, 0)
        out = np.zeros(diag_arr.shape[:-1], dtype=a.dtype)
        common_shape_impl(diag_arr, out, dpnp_func, PRINT_DEBUG)
        return out

    return dpnp_impl
