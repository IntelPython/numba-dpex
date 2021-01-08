import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba import types
from numba.core.typing import signature
from . import stubs
import numba_dppy.dpnp_glue as dpnp_lowering
from numba.core.extending import overload, register_jitable
import numpy as np
from numba_dppy import dpctl_functions
import os


@register_jitable
def check_range(low, high):
    if low >= high:
        raise ValueError("Low cannot be >= High")

@register_jitable
def common_impl(low, high, res, dpnp_func, PRINT_DEBUG):
    check_range(low, high)

    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    dpnp_func(res_usm, low, high, res.size)

    dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if PRINT_DEBUG:
        print("DPNP implementation")


@overload(stubs.dpnp.random)
@overload(stubs.dpnp.sample)
@overload(stubs.dpnp.ranf)
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

        common_impl(0, 1, res, dpnp_func, PRINT_DEBUG)
        return res

    return dpnp_impl

@overload(stubs.dpnp.rand)
def dpnp_random_sample(*size):
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

    def dpnp_impl(*size):
        res = np.empty(size, dtype=res_dtype)

        for i in size:
            if i == 0:
                return res

        common_impl(0, 1, res, dpnp_func, PRINT_DEBUG)
        return res

    return dpnp_impl


@overload(stubs.dpnp.randint)
def dpnp_random_sample(low, high=None, size=None):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp("randint")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):
        if high not in (None, types.none):
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(low, high, res, dpnp_func, PRINT_DEBUG)
                return res
        else:
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(0, low, res, dpnp_func, PRINT_DEBUG)
                return res
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        if high not in (None, types.none):
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(low, high, res, dpnp_func, PRINT_DEBUG)
                return res
        else:
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(0, low, res, dpnp_func, PRINT_DEBUG)
                return res

    return dpnp_impl


@overload(stubs.dpnp.random_integers)
def dpnp_random_sample(low, high=None, size=None):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp("random_integers")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):
        if high not in (None, types.none):
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(low, high+1, res, dpnp_func, PRINT_DEBUG)
                return res
        else:
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(1, low+1, res, dpnp_func, PRINT_DEBUG)
                return res
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        if high not in (None, types.none):
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(low, high+1, res, dpnp_func, PRINT_DEBUG)
                return res
        else:
            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(1, low+1, res, dpnp_func, PRINT_DEBUG)
                return res

    return dpnp_impl


@overload(stubs.dpnp.beta)
def dpnp_random_sample(a, b, size=None):
    name = "beta"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L36

    Function declaration:
    void custom_rng_beta_c(void* result, _DataType a, _DataType b, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(a, b, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, a, b, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not (isinstance(a, types.Float)):
        raise ValueError("We only support float scalar for input: a")

    if not (isinstance(b, types.Float)):
        raise ValueError("We only support float scalar for input: b")

    if size in (None, types.none):
        def dpnp_impl(a, b, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(a, b, res)
            return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        def dpnp_impl(a, b, size=None):
            res = np.empty(size, dtype=res_dtype)
            if t:
                for i in size:
                    if i == 0:
                        return res
            else:
                if size == 0:
                    return res
            common_impl(a, b, res)
            return res

    return dpnp_impl


@overload(stubs.dpnp.binomial)
def dpnp_random_sample(n, p, size=None):
    name = "binomial"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L56

    Function declaration:
    void custom_rng_binomial_c(void* result, int ntrial, double p, size_t size)

    """
    sig = signature(
        ret_type, types.voidptr, types.int32, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(n, p, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, n, p, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not (isinstance(n, types.Integer)):
        raise ValueError("We only support scalar for input: n")

    if not (isinstance(p, types.Float)):
        raise ValueError("We only support scalar for input: p")

    if size in (None, types.none):
        def dpnp_impl(n, p, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(n, p, res)
            return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        def dpnp_impl(n, p, size=None):
            res = np.empty(size, dtype=res_dtype)
            if t:
                for i in size:
                    if i == 0:
                        return res
            else:
                if size == 0:
                    return res
            common_impl(n, p, res)
            return res

    return dpnp_impl


@overload(stubs.dpnp.chisquare)
def dpnp_random_sample(df, size=None):
    name = "chisquare"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L71

    Function declaration:
    void custom_rng_chi_square_c(void* result, int df, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.int32, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(df, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, df, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not (isinstance(df, types.Integer)):
        raise ValueError("We only support scalar for input: df")

    if size in (None, types.none):
        def dpnp_impl(df, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(df, res)
            return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        def dpnp_impl(df, size=None):
            res = np.empty(size, dtype=res_dtype)
            if t:
                for i in size:
                    if i == 0:
                        return res
            else:
                if size == 0:
                    return res
            common_impl(df, res)
            return res

    return dpnp_impl


@overload(stubs.dpnp.exponential)
def dpnp_random_sample(scale=1.0, size=None):
    name = "exponential"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L86

    Function declaration:
    void custom_rng_exponential_c(void* result, _DataType beta, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(scale, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, scale, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):
        if isinstance(scale, float):
            def dpnp_impl(scale=1.0, size=None):
                scale = 1.0
                res = np.empty(1, dtype=res_dtype)
                common_impl(scale, res)
                return res[0]
        else:
            def dpnp_impl(scale=1.0, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(scale, res)
                return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False
        if isinstance(scale, float):
            def dpnp_impl(scale=1.0, size=None):
                scale = 1.0
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(scale, res)
                return res
        else:
            def dpnp_impl(scale=1.0, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(scale, res)

    return dpnp_impl


@overload(stubs.dpnp.gamma)
def dpnp_random_sample(shape, scale=1.0, size=None):
    name = "gamma"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L105

    Function declaration:
    void custom_rng_gamma_c(void* result, _DataType shape, _DataType scale, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(shape, scale, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, shape, scale, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not (isinstance(shape, types.Float)):
        raise ValueError("We only support scalar for input: shape")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):
        if isinstance(scale, float):
            def dpnp_impl(shape, scale=1.0, size=None):
                scale = 1.0
                res = np.empty(1, dtype=res_dtype)
                common_impl(shape, scale, res)
                return res[0]
        else:
            def dpnp_impl(shape, scale=1.0, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(shape, scale, res)
                return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False
        if isinstance(scale, float):
            def dpnp_impl(shape, scale=1.0, size=None):
                scale = 1.0
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(shape, scale, res)
                return res
        else:
            def dpnp_impl(shape, scale=1.0, size=None):
                res = np.empty(size, dtype=res_dtype)
                if t:
                    for i in size:
                        if i == 0:
                            return res
                else:
                    if size == 0:
                        return res
                common_impl(shape, scale, res)

    return dpnp_impl


@overload(stubs.dpnp.geometric)
def dpnp_random_sample(p, size=None):
    name = "geometric"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L139

    Function declaration:
    void custom_rng_geometric_c(void* result, float p, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.float32, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_"+name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    @register_jitable
    def common_impl(p, res):
        sycl_queue = dpctl_functions.get_current_queue()
        res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

        dpnp_func(res_usm, p, res.size)

        dpctl_functions.queue_memcpy(sycl_queue, res.ctypes, res_usm, res.size * res.itemsize)

        dpctl_functions.free_with_queue(res_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([res.size])

        if PRINT_DEBUG:
            print("DPNP implementation")

    if not (isinstance(p, types.Float)):
        raise ValueError("We only support scalar for input: p")

    if size in (None, types.none):
        def dpnp_impl(p, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(p, res)
            return res[0]
    else:
        if isinstance(size, types.UniTuple):
            t = True
        else:
            t = False

        def dpnp_impl(p, size=None):
            res = np.empty(size, dtype=res_dtype)
            if t:
                for i in size:
                    if i == 0:
                        return res
            else:
                if size == 0:
                    return res
            common_impl(p, res)
            return res

    return dpnp_impl


