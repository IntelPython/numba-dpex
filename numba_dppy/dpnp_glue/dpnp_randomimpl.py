import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy.dpctl_functions import _DPCTL_FUNCTIONS


@overload(stubs.dpnp.random_sample)
def dpnp_random_sample(size):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64

    def dpnp_impl(size):
        res = np.empty(size, dtype=res_dtype)

        for i in size:
            if i == 0:
                return res

        sycl_queue = get_sycl_queue()
        res_usm = allocate_usm_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, 0, 1, res.size)

        copy_usm(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        free_usm(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        return res

    return dpnp_impl


