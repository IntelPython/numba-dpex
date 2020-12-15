import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba.core import types, cgutils
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np


@overload(stubs.dpnp.max)
@overload(stubs.dpnp.amax)
def dpnp_amax_impl(a):
    name = "max"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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
    sig = signature(ret_type, types.voidptr, types.voidptr,
                              types.voidptr, types.intp,
                              types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = allocate_usm_shared(a.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        out = np.empty(1, dtype=a.dtype)
        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size])

        print("DDD")
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.min)
@overload(stubs.dpnp.amin)
def dpnp_amin_impl(a):
    name = "min"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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
    sig = signature(ret_type, types.voidptr, types.voidptr,
                              types.voidptr, types.intp,
                              types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = allocate_usm_shared(a.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, 0)

        out = np.empty(1, dtype=a.dtype)
        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size])

        print("EEE")
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.mean)
def dpnp_mean_impl(a):
    name = "mean"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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
    sig = signature(ret_type, types.voidptr, types.voidptr,
                              types.voidptr, types.intp,
                              types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = allocate_usm_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])
        print("FFFF")
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.median)
def dpnp_median_impl(a):
    name = "median"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

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
    sig = signature(ret_type, types.voidptr, types.voidptr,
                              types.voidptr, types.intp,
                              types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        a_usm = allocate_usm_shared(a.size * a.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out = np.empty(1, dtype=res_dtype)
        out_usm = allocate_usm_shared(out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim, a.shapeptr, a.ndim)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a.size, out.size])

        print("GG")
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.cov)
def dpnp_cov_impl(a):
    name = "cov"
    dpnp_lowering.ensure_dpnp(name)
    dpctl_functions = dpnp_ext._DPCTL_FUNCTIONS()

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_statistics.cpp#L51

    Function declaration:
    void custom_cov_c(void* array1_in, void* result1, size_t nrows, size_t ncols)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr,
                              types.intp, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, [a.dtype.name, "NONE"], sig)

    get_sycl_queue = dpctl_functions.dpctl_get_current_queue()
    allocate_usm_shared = dpctl_functions.dpctl_malloc_shared()
    copy_usm = dpctl_functions.dpctl_queue_memcpy()
    free_usm = dpctl_functions.dpctl_free_with_queue()

    res_dtype = np.float64
    copy_input_to_double = True
    if a.dtype == types.float64:
        copy_input_to_double = False


    def dpnp_impl(a):
        if a.size == 0:
            raise ValueError("Passed Empty array")

        sycl_queue = get_sycl_queue()

        """ We have to pass a array in double precision to DpNp """
        if copy_input_to_double:
            a_copy_in_double = a.astype(np.float64)
        else:
            a_copy_in_double = a
        a_usm = allocate_usm_shared(a_copy_in_double.size * a_copy_in_double.itemsize, sycl_queue)
        copy_usm(sycl_queue, a_usm, a_copy_in_double.ctypes,
                 a_copy_in_double.size * a_copy_in_double.itemsize)

        if a.ndim == 2:
            rows = a.shape[0]
            cols = a.shape[1]
            out = np.empty((rows, rows), dtype=res_dtype)
        elif a.ndim == 1:
            rows = 1
            cols = a.shape[0]
            out = np.empty(rows, dtype=res_dtype)

        out_usm = allocate_usm_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, rows, cols)

        copy_usm(sycl_queue, out.ctypes, out_usm, out.size * out.itemsize)

        free_usm(a_usm, sycl_queue)
        free_usm(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([a_copy_in_double.size, a.size, out.size])

        print("KK")
        if a.ndim == 2:
            return out
        elif a.ndim == 1:
            return out[0]

    return dpnp_impl


