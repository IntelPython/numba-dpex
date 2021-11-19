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

import numba_dppy.dpnp_glue as dpnp_lowering
import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba_dppy import dpctl_functions

from . import stubs


@overload(stubs.dpnp.diagonal)
def dpnp_diagonal_impl(a, offset=0):
    name = "diagonal"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/e389248c709531b181be8bf33b1a270fca812a92/dpnp/backend/kernels/dpnp_krnl_indexing.cpp#L39

    Function declaration:
    void dpnp_diagonal_c(
        void* array1_in, const size_t input1_size, void* result1, const size_t offset, size_t* shape, size_t* res_shape, const size_t res_ndim)

    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.voidptr,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    function_text = f"""\
def tuplizer(a):
    return ({", ".join(f"a[{i}]" for i in range(a.ndim - 1))})
"""
    locals = {}
    exec(function_text, globals(), locals)
    tuplizer = register_jitable(locals["tuplizer"])

    def dpnp_impl(a, offset=0):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        n = min(a.shape[0], a.shape[1])
        res_shape = np.zeros(a.ndim - 1, dtype=np.int64)

        if a.ndim > 2:
            for i in range(a.ndim - 2):
                res_shape[i] = a.shape[i + 2]

        if (n + offset) > a.shape[1]:
            res_shape[-1] = a.shape[1] - offset
        elif (n + offset) > a.shape[0]:
            res_shape[-1] = a.shape[0]
        else:
            res_shape[-1] = n + offset

        shape = tuplizer(res_shape)

        out = np.empty(shape, dtype=a.dtype)

        sycl_queue = dpctl_functions.get_current_queue()

        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out_usm = dpctl_functions.malloc_shared(
            out.size * out.itemsize, sycl_queue
        )

        dpnp_func(
            a_usm,
            a.size * a.itemsize,
            out_usm,
            offset,
            a.shapeptr,
            out.shapeptr,
            out.ndim,
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
