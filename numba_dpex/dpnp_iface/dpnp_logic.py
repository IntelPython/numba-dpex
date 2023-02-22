# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numba import types
from numba.core.extending import overload
from numba.core.typing import signature

import numba_dpex.dpctl_iface as dpctl_functions
import numba_dpex.dpnp_iface as dpnp_lowering
import numba_dpex.dpnp_iface.dpnpimpl as dpnp_ext

from . import stubs


@overload(stubs.dpnp.all)
def dpnp_all_impl(a):
    name = "all"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.6.2/dpnp/backend/kernels/dpnp_krnl_logic.cpp#L36
    Function declaration:
    void dpnp_all_c(const void* array1_in, void* result1, const size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        if a.size == 0:
            return True

        out = np.empty(1, dtype=np.bool_)

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

        # TODO: sometimes all() returns ndarray
        return out[0]

    return dpnp_impl
