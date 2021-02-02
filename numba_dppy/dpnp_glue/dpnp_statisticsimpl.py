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

import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba.core import types, cgutils
from numba.core.typing import signature
from numba_dppy.numpy import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy import dpctl_functions


@overload(stubs.numpy.max)
@overload(stubs.numpy.amax)
def dpnp_amax_impl(a):
    name = "max"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L129

    Function declaration:
    void custom_max_c(void* array1_in, void* result1, const size_t* shape,
                      size_t ndim, const size_t* axis, size_t naxis)

    We are using void * in case of size_t * as Numba currently does not have
    any type to represent size_t *. Since, both the types are pointers,
    if the compiler allows there should not be any mismatch in the size of
    the container to hold different types of pointer.
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        out = np.empty(1, dtype=a.dtype)
        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.min)
@overload(stubs.numpy.amin)
def dpnp_amin_impl(a):
    name = "min"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L247

    Function declaration:
    void custom_min_c(void* array1_in, void* result1, const size_t* shape,
                      size_t ndim, const size_t* axis, size_t naxis)

    We are using void * in case of size_t * as Numba currently does not have
    any type to represent size_t *. Since, both the types are pointers,
    if the compiler allows there should not be any mismatch in the size of
    the container to hold different types of pointer.
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, 0)

        out = np.empty(1, dtype=a.dtype)
        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.mean)
def dpnp_mean_impl(a):
    name = "mean"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L169

    Function declaration:
    void custom_mean_c(void* array1_in, void* result1, const size_t* shape,
                       size_t ndim, const size_t* axis, size_t naxis)

    We are using void * in case of size_t * as Numba currently does not have
    any type to represent size_t *. Since, both the types are pointers,
    if the compiler allows there should not be any mismatch in the size of
    the container to hold different types of pointer.
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])
        return out[0]

    return dpnp_impl


@overload(stubs.numpy.median)
def dpnp_median_impl(a):
    name = "median"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L213

    Function declaration:
    void custom_median_c(void* array1_in, void* result1, const size_t* shape,
			 size_t ndim, const size_t* axis, size_t naxis)

    We are using void * in case of size_t * as Numba currently does not have
    any type to represent size_t *. Since, both the types are pointers,
    if the compiler allows there should not be any mismatch in the size of
    the container to hold different types of pointer.
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.cov)
def dpnp_cov_impl(a):
    name = "cov"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L51

    Function declaration:
    void custom_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = np.float64
    copy_input_to_double = True
    if a.dtype == types.float64:
        copy_input_to_double = False

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        """ We have to pass a array in double precision to DpNp """
        if copy_input_to_double:
            a_copy_in_double = a.astype(np.float64)
        else:
            a_copy_in_double = a
        a_usm = dpctl_functions.malloc_shared(
            a_copy_in_double.size * a_copy_in_double.itemsize, sycl_queue
        )
        dpctl_functions.queue_memcpy(
            sycl_queue,
            a_usm,
            a_copy_in_double.ctypes,
            a_copy_in_double.size * a_copy_in_double.itemsize,
        )

        if a.ndim == 2:
            rows = a.shape[0]
            cols = a.shape[1]
            out = np.empty((rows, rows), dtype=res_dtype)
        elif a.ndim == 1:
            rows = 1
            cols = a.shape[0]
            out = np.empty(rows, dtype=res_dtype)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, rows, cols)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a_copy_in_double.size, a.size, out.size])

        if a.ndim == 2:
            return out
        elif a.ndim == 1:
            return out[0]

    return dpnp_impl
