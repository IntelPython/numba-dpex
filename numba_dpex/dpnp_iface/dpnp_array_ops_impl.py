# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import types
from numba.core.extending import overload, register_jitable
from numba.core.typing import signature

import numba_dpex
import numba_dpex.dpctl_iface as dpctl_functions
import numba_dpex.dpnp_iface as dpnp_lowering
import numba_dpex.dpnp_iface.dpnp_stubs_impl as dpnp_ext

from . import stubs


@register_jitable
def common_impl(a, out, dpnp_func, print_debug):
    if a.size == 0:
        raise ValueError("Passed Empty array")

    sycl_queue = dpctl_functions.get_current_queue()
    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)

    dpnp_func(a_usm, out_usm, a.size)

    event = dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, out.size])

    if print_debug:
        print("dpnp implementation")


@overload(stubs.dpnp.cumsum)
def dpnp_cumsum_impl(a):
    name = "cumsum"
    dpnp_lowering.ensure_dpnp(name)

    res_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_mathematical.cpp#L135
    Function declaration:
    void dpnp_cumsum_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(res_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        out = np.arange(0, a.size, 1, a.dtype)
        common_impl(a, out, dpnp_func, PRINT_DEBUG)

        return out

    return dpnp_impl


@overload(stubs.dpnp.cumprod)
def dpnp_cumprod_impl(a):
    name = "cumprod"
    dpnp_lowering.ensure_dpnp(name)

    res_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.1/dpnp/backend/kernels/dpnp_krnl_mathematical.cpp#L110
    Function declaration:
    void dpnp_cumprod_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(res_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG
    if a.dtype == types.Integer:
        ret_dtype = np.int64
    else:
        ret_dtype = a.dtype

    def dpnp_impl(a):
        out = np.arange(0, a.size, 1, ret_dtype)
        common_impl(a, out, dpnp_func, PRINT_DEBUG)

        return out

    return dpnp_impl


@overload(stubs.dpnp.copy)
def dpnp_copy_impl(a):
    name = "copy"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.10.0dev0/dpnp/backend/kernels/dpnp_krnl_elemwise.cpp#L320-L330
    Function declaration:
    void __name__(void* result_out,                                                                                     \
                  const size_t result_size,                                                                             \
                  const size_t result_ndim,                                                                             \
                  const shape_elem_type* result_shape,                                                                  \
                  const shape_elem_type* result_strides,                                                                \
                  const void* input1_in,                                                                                \
                  const size_t input1_size,                                                                             \
                  const size_t input1_ndim,                                                                             \
                  const shape_elem_type* input1_shape,                                                                  \
                  const shape_elem_type* input1_strides,                                                                \
                  const size_t* where);
    """
    sig = signature(
        ret_type,
        types.voidptr,  # void* result_out
        types.intp,  # const size_t result_size,
        types.intp,  # const size_t result_ndim,
        types.voidptr,  # const shape_elem_type* result_shape,
        types.voidptr,  # const shape_elem_type* result_strides,
        types.voidptr,  # const void* input1_in,
        types.intp,  # const size_t input1_size,
        types.intp,  # const size_t input1_ndim,
        types.voidptr,  # const shape_elem_type* input1_shape,
        types.voidptr,  # const shape_elem_type* input1_strides,
        types.voidptr,  # const size_t* where);
    )

    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = a.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out = np.arange(0, a.size, 1, res_dtype)
        out_usm = dpctl_functions.malloc_shared(
            out.size * out.itemsize, sycl_queue
        )

        strides = np.array(1)

        result_out = out_usm
        result_size = out.size
        result_ndim = out.ndim
        result_shape = out.shapeptr
        result_strides = strides.ctypes

        input1_in = a_usm
        input1_size = a.size
        input1_ndim = a.ndim
        input1_shape = a.shapeptr
        input1_strides = strides.ctypes

        where = 0

        dpnp_func(
            result_out,
            result_size,
            result_ndim,
            result_shape,
            result_strides,
            input1_in,
            input1_size,
            input1_ndim,
            input1_shape,
            input1_strides,
            where,
        )

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
        return out

    return dpnp_impl


@overload(stubs.dpnp.sort)
def dpnp_sort_impl(a):
    name = "sort"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.5.0/dpnp/backend/kernels/dpnp_krnl_sorting.cpp#L90

    Function declaration:
    void dpnp_sort_c(void* array1_in, void* result1, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = a.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out = np.arange(0, a.size, 1, res_dtype)
        out_usm = dpctl_functions.malloc_shared(
            out.size * out.itemsize, sycl_queue
        )

        dpnp_func(a_usm, out_usm, a.size)

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
        return out

    return dpnp_impl


@overload(stubs.dpnp.take)
def dpnp_take_impl(a, ind):
    name = "take"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/9b14f0ca76a9e0c309bb97b4d5caa0870eecd6bb/dpnp/backend/kernels/dpnp_krnl_indexing.cpp#L925
    Function declaration:
    void dpnp_take_c(void* array1_in, const size_t array1_size, void* indices1, void* result1, size_t size)
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func(
        "dpnp_" + name, [a.dtype.name, ind.dtype.name], sig
    )

    res_dtype = a.dtype
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, ind):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        ind_usm = dpctl_functions.malloc_shared(
            ind.size * ind.itemsize, sycl_queue
        )
        event = dpctl_functions.queue_memcpy(
            sycl_queue, ind_usm, ind.ctypes, ind.size * ind.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out = np.arange(0, ind.size, 1, res_dtype).reshape(ind.shape)
        out_usm = dpctl_functions.malloc_shared(
            out.size * out.itemsize, sycl_queue
        )

        dpnp_func(a_usm, a.size * a.itemsize, ind_usm, out_usm, ind.size)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(ind_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, ind.size, out.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return out

    return dpnp_impl
