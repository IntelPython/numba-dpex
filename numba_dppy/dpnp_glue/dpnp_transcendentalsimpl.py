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
def common_impl(a, out, dpnp_func, PRINT_DEBUG):
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

    dpnp_ext._dummy_liveness_func([out.size])

    if PRINT_DEBUG:
        print("DPNP implementation")


@overload(stubs.dpnp.sum)
def dpnp_sum_impl(a):
    name = "sum"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_reduction.cpp#L39

    Function declaration:
    void custom_sum_c(void* array1_in, void* result1, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
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

    def dpnp_impl(a):
        a_copy = a.copy()
        a_copy = np.ravel(a_copy)

        for i in range(len(a_copy)):
            if np.isnan(a_copy[i]):
                a_copy[i] = 0

        result = numba_dppy.dpnp.sum(a_copy)
        dpnp_ext._dummy_liveness_func([a_copy.size])
        return result

    return dpnp_impl


@overload(stubs.dpnp.nanprod)
def dpnp_nanprod_impl(a):
    name = "nanprod"
    dpnp_lowering.ensure_dpnp(name)

    def dpnp_impl(a):
        a_copy = a.copy()
        a_copy = np.ravel(a_copy)

        for i in range(len(a_copy)):
            if np.isnan(a_copy[i]):
                a_copy[i] = 1

        result = numba_dppy.dpnp.prod(a_copy)
        dpnp_ext._dummy_liveness_func([a_copy.size])
        return result

    return dpnp_impl
