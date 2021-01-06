import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy import dpctl_functions
import os


@overload(stubs.dpnp.random_sample)
def dpnp_random_sample(size):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG
    if isinstance(size, types.UniTuple):
        t = True
    else:
        t = False

    def dpnp_impl(size):
        res = np.empty(size, dtype=res_dtype)

        if t:
            for i in size:
                if i == 0:
                    return res
        else:
            if size == 0:
                return res

        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, 0, 1, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

        return res

    return dpnp_impl


