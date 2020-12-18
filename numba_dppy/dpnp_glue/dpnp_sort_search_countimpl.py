import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba.core import types, cgutils
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np


@overload(stubs.dpnp.argmax)
def dpnp_argmax_impl(a):
    name = "argmax"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L36

    Function declaration:
    void custom_argmax_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, np.dtype(np.int64).name], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = allocate_usm_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.argmin)
def dpnp_argmin_impl(a):
    name = "argmin"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L56

    Function declaration:
    void custom_argmin_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, np.dtype(np.int64).name], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = allocate_usm_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.argsort)
def dpnp_argsort_impl(a):
    name = "argsort"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_searching.cpp#L56

    Function declaration:
    void custom_argmin_c(void* array1_in, void* result1, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.int64

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.arange(a.size, dtype=res_dtype)
        out_usm = allocate_usm_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.size)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        return out

    return dpnp_impl
