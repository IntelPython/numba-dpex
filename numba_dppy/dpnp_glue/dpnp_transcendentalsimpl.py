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
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy import dpctl_functions
import numba_dppy


@register_jitable
def common_impl(a, out, dpnp_func, print_debug):
    if a.size == 0:
        raise ValueError("Passed Empty array")

    sycl_queue = dpctl_functions.get_current_queue()
    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

    out_usm = dpctl_functions.malloc_shared(a.itemsize, sycl_queue)

    initial = np.zeros(1, dtype=a.dtype)
    axes, axes_ndim = 0, 0
    where = 0

    dpnp_func(out_usm, a_usm, a.shapeptr, a.ndim, axes, axes_ndim, initial.ctypes, where)

    dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, out.size])

    if print_debug:
        print("dpnp implementation")


@overload(stubs.dpnp.sum)
def dpnp_sum_impl(a):
    name = "sum"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.6.1dev/dpnp/backend/kernels/dpnp_krnl_reduction.cpp#L59

    Function declaration:
    void dpnp_sum_c(void* result_out,
                    const void* input_in,
                    const size_t* input_shape,
                    const size_t input_shape_ndim,
                    const long* axes,
                    const size_t axes_ndim,
                    const void* initial,
                    const long* where)

    """
    sig = signature(ret_type,
                    types.voidptr, types.voidptr,
                    types.voidptr, types.intp,
                    types.voidptr, types.intp,
                    types.voidptr, types.voidptr)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        out = np.empty(1, dtype=a.dtype)
        common_impl(a, out, dpnp_func, PRINT_DEBUG)

        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.prod)
def dpnp_prod_impl(a):
    name = "prod"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_reduction.cpp#L83

    Function declaration:
    void custom_prod_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        out = np.empty(1, dtype=a.dtype)

        common_impl(a, out, dpnp_func, PRINT_DEBUG)
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.nansum)
def dpnp_nansum_impl(a):
    name = "nansum"
    dpnp_lowering.ensure_dpnp(name)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        a_copy = a.copy()
        a_copy = np.ravel(a_copy)

        for i in range(len(a_copy)):
            if np.isnan(a_copy[i]):
                a_copy[i] = 0

        result = numba_dppy.dpnp.sum(a_copy)
        dpnp_ext._dummy_liveness_func([a_copy.size])

        if PRINT_DEBUG:
            print("dpnp implementation")

        return result

    return dpnp_impl


@overload(stubs.dpnp.nanprod)
def dpnp_nanprod_impl(a):
    name = "nanprod"
    dpnp_lowering.ensure_dpnp(name)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        a_copy = a.copy()
        a_copy = np.ravel(a_copy)

        for i in range(len(a_copy)):
            if np.isnan(a_copy[i]):
                a_copy[i] = 1

        result = numba_dppy.dpnp.prod(a_copy)
        dpnp_ext._dummy_liveness_func([a_copy.size])

        if PRINT_DEBUG:
            print("dpnp implementation")

        return result

    return dpnp_impl
