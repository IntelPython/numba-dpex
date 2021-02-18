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

    dpnp_func(a_usm, out_usm, a.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )

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
    https://github.com/IntelPython/dpnp/blob/master/dpnp/backend/kernels/dpnp_krnl_mathematical.cpp#L135
    Function declaration:
    void dpnp_cumsum_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(res_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        out = np.arange(a.size, dtype=a.dtype)
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
    https://github.com/IntelPython/dpnp/blob/master/dpnp/backend/kernels/dpnp_krnl_mathematical.cpp#L110
    Function declaration:
    void dpnp_cumprod_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(res_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        out = np.arange(a.size, dtype=a.dtype)
        common_impl(a, out, dpnp_func, PRINT_DEBUG)

        return out

    return dpnp_impl
