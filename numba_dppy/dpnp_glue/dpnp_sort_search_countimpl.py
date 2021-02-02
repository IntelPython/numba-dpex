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


@overload(stubs.numpy.argmax)
def dpnp_argmax_impl(a):
    name = "argmax"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L36

    Function declaration:
    void custom_argmax_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func(
        "dpnp_" + name, [a.dtype.name, np.dtype(np.int64).name], sig
    )

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.argmin)
def dpnp_argmin_impl(a):
    name = "argmin"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L56

    Function declaration:
    void custom_argmin_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func(
        "dpnp_" + name, [a.dtype.name, np.dtype(np.int64).name], sig
    )

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.argsort)
def dpnp_argsort_impl(a):
    name = "argsort"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L56

    Function declaration:
    void custom_argmin_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.arange(a.size, dtype=res_dtype)
        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out

    return dpnp_impl
