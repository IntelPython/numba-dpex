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
from numba_dppy.numpy import stubs
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
def common_impl(low, high, res, dpnp_func, print_debug):
    check_range(low, high)

    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    dpnp_func(res_usm, low, high, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )
    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_0_arg(res, dpnp_func, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    dpnp_func(res_usm, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )
    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_1_arg(arg1, res, dpnp_func, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    try:
        dpnp_func(res_usm, arg1, res.size)
    except Exception:
        raise ValueError("Device not supported")

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )
    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_2_arg(arg1, arg2, res, dpnp_func, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    dpnp_func(res_usm, arg1, arg2, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )
    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_hypergeometric(ngood, nbad, nsample, res, dpnp_func, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    dpnp_func(res_usm, ngood, nbad, nsample, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )

    dpctl_functions.free_with_queue(res_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_multinomial(n, pvals, res, dpnp_func, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    pvals_usm = dpctl_functions.malloc_shared(pvals.size * pvals.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(
        sycl_queue, pvals_usm, pvals.ctypes, pvals.size * pvals.itemsize
    )

    dpnp_func(res_usm, n, pvals_usm, pvals.size, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )

    dpctl_functions.free_with_queue(res_usm, sycl_queue)
    dpctl_functions.free_with_queue(pvals_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_impl_multivariate_normal(
    mean, cov, size, check_valid, tol, res, dpnp_func, print_debug
):
    sycl_queue = dpctl_functions.get_current_queue()
    res_usm = dpctl_functions.malloc_shared(res.size * res.itemsize, sycl_queue)

    mean_usm = dpctl_functions.malloc_shared(mean.size * mean.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(
        sycl_queue, mean_usm, mean.ctypes, mean.size * mean.itemsize
    )

    cov_usm = dpctl_functions.malloc_shared(cov.size * cov.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(
        sycl_queue, cov_usm, cov.ctypes, cov.size * cov.itemsize
    )

    dpnp_func(res_usm, mean.size, mean_usm, mean.size, cov_usm, cov.size, res.size)

    dpctl_functions.queue_memcpy(
        sycl_queue, res.ctypes, res_usm, res.size * res.itemsize
    )

    dpctl_functions.free_with_queue(res_usm, sycl_queue)
    dpctl_functions.free_with_queue(mean_usm, sycl_queue)
    dpctl_functions.free_with_queue(cov_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([res.size])

    if print_debug:
        print("dpnp implementation")


@overload(stubs.numpy.random)
@overload(stubs.numpy.sample)
@overload(stubs.numpy.ranf)
@overload(stubs.numpy.random_sample)
def dpnp_random_impl(size):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(size):
        res = np.empty(size, dtype=res_dtype)
        if res.size != 0:
            common_impl(0, 1, res, dpnp_func, PRINT_DEBUG)
        return res

    return dpnp_impl


@overload(stubs.numpy.rand)
def dpnp_random_impl(*size):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(*size):
        res = np.empty(size, dtype=res_dtype)
        if res.size != 0:
            common_impl(0, 1, res, dpnp_func, PRINT_DEBUG)
        return res

    return dpnp_impl


@overload(stubs.numpy.randint)
def dpnp_random_impl(low, high=None, size=None):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp("randint")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)

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
        if high not in (None, types.none):

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if res.size != 0:
                    common_impl(low, high, res, dpnp_func, PRINT_DEBUG)
                return res

        else:

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if res.size != 0:
                    common_impl(0, low, res, dpnp_func, PRINT_DEBUG)
                return res

    return dpnp_impl


@overload(stubs.numpy.random_integers)
def dpnp_random_impl(low, high=None, size=None):
    name = "random_sample"
    dpnp_lowering.ensure_dpnp("random_integers")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):
        if high not in (None, types.none):

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(low, high + 1, res, dpnp_func, PRINT_DEBUG)
                return res

        else:

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(1, dtype=res_dtype)
                common_impl(1, low + 1, res, dpnp_func, PRINT_DEBUG)
                return res

    else:
        if high not in (None, types.none):

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if res.size != 0:
                    common_impl(low, high + 1, res, dpnp_func, PRINT_DEBUG)
                return res

        else:

            def dpnp_impl(low, high=None, size=None):
                res = np.empty(size, dtype=res_dtype)
                if res.size != 0:
                    common_impl(1, low + 1, res, dpnp_func, PRINT_DEBUG)
                return res

    return dpnp_impl


@overload(stubs.numpy.beta)
def dpnp_random_impl(a, b, size=None):
    name = "beta"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L36

    Function declaration:
    void custom_rng_beta_c(void* result, _DataType a, _DataType b, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(a, types.Float)):
        raise ValueError("We only support float scalar for input: a")

    if not (isinstance(b, types.Float)):
        raise ValueError("We only support float scalar for input: b")

    if size in (None, types.none):

        def dpnp_impl(a, b, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(a, b, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(a, b, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(a, b, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.binomial)
def dpnp_random_impl(n, p, size=None):
    name = "binomial"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L56

    Function declaration:
    void custom_rng_binomial_c(void* result, int ntrial, double p, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.int32, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)
    res_dtype = np.int32
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(n, types.Integer)):
        raise ValueError("We only support scalar for input: n")

    if not (isinstance(p, types.Float)):
        raise ValueError("We only support scalar for input: p")

    if size in (None, types.none):

        def dpnp_impl(n, p, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(n, p, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(n, p, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(n, p, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.chisquare)
def dpnp_random_impl(df, size=None):
    name = "chisquare"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L71

    Function declaration:
    void custom_rng_chi_square_c(void* result, int df, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.int32, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(df, types.Integer)):
        raise ValueError("We only support scalar for input: df")

    if size in (None, types.none):

        def dpnp_impl(df, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(df, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(df, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(df, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.exponential)
def dpnp_random_impl(scale=1.0, size=None):
    name = "exponential"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L86

    Function declaration:
    void custom_rng_exponential_c(void* result, _DataType beta, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(scale, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.gamma)
def dpnp_random_impl(shape, scale=1.0, size=None):
    name = "gamma"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L105

    Function declaration:
    void custom_rng_gamma_c(void* result, _DataType shape, _DataType scale, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(shape, types.Float)):
        raise ValueError("We only support scalar for input: shape")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(shape, scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(shape, scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(shape, scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(shape, scale, res, dpnp_func, PRINT_DEBUG)

    return dpnp_impl


@overload(stubs.numpy.geometric)
def dpnp_random_impl(p, size=None):
    name = "geometric"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L139

    Function declaration:
    void custom_rng_geometric_c(void* result, float p, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float32, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)
    res_dtype = np.int32
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(p, types.Float)):
        raise ValueError("We only support scalar for input: p")

    if size in (None, types.none):

        def dpnp_impl(p, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(p, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(p, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(p, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.gumbel)
def dpnp_random_impl(loc=0.0, scale=1.0, size=None):
    name = "gumbel"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L154

    Function declaration:
    void custom_rng_gumbel_c(void* result, double loc, double scale, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(loc, float):
        if not (isinstance(loc, types.Float)):
            raise ValueError("We only support scalar for input: loc")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.hypergeometric)
def dpnp_random_impl(ngood, nbad, nsample, size=None):
    name = "hypergeometric"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L169

    Function declaration:
    void custom_rng_hypergeometric_c(void* result, int l, int s, int m, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.int32, types.int32, types.int32, types.intp
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(ngood, types.Integer)):
        raise ValueError("We only support scalar for input: ngood")

    if not (isinstance(nbad, types.Integer)):
        raise ValueError("We only support scalar for input: nbad")

    if not (isinstance(nsample, types.Integer)):
        raise ValueError("We only support scalar for input: nsample")

    if size in (None, types.none):

        def dpnp_impl(ngood, nbad, nsample, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_hypergeometric(
                ngood, nbad, nsample, res, dpnp_func, PRINT_DEBUG
            )
            return res[0]

    else:

        def dpnp_impl(ngood, nbad, nsample, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_hypergeometric(
                    ngood, nbad, nsample, res, dpnp_func, PRINT_DEBUG
                )
            return res

    return dpnp_impl


@overload(stubs.numpy.laplace)
def dpnp_random_impl(loc=0.0, scale=1.0, size=None):
    name = "laplace"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L184

    Function declaration:
    void custom_rng_laplace_c(void* result, double loc, double scale, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(loc, float):
        if not (isinstance(loc, types.Float)):
            raise ValueError("We only support scalar for input: loc")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.lognormal)
def dpnp_random_impl(mean=0.0, sigma=1.0, size=None):
    name = "lognormal"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L199

    Function declaration:
    void custom_rng_lognormal_c(void* result, _DataType mean, _DataType stddev, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(mean, float):
        if not (isinstance(mean, types.Float)):
            raise ValueError("We only support scalar for input: loc")

    if not isinstance(sigma, float):
        if not (isinstance(sigma, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(mean=0.0, sigma=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(mean, sigma, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(mean=0.0, sigma=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(mean, sigma, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.multinomial)
def dpnp_random_impl(n, pvals, size=None):
    name = "multinomial"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L218

    Function declaration:
    void custom_rng_multinomial_c(void* result, int ntrial, const double* p_vector,
                                  const size_t p_vector_size, size_t size)
    """
    sig = signature(
        ret_type, types.voidptr, types.int32, types.voidptr, types.intp, types.intp
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)

    res_dtype = np.int32

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(n, types.Integer):
        raise TypeError(
            "np.random.multinomial(): n should be an " "integer, got %s" % (n,)
        )

    if not isinstance(pvals, (types.Sequence, types.Array)):
        raise TypeError(
            "np.random.multinomial(): pvals should be an "
            "array or sequence, got %s" % (pvals,)
        )

    if size in (None, types.none):

        def dpnp_impl(n, pvals, size=None):
            out = np.zeros(len(pvals), res_dtype)
            common_impl_multinomial(n, pvals, out, dpnp_func, PRINT_DEBUG)
            return out

    elif isinstance(size, types.Integer):

        def dpnp_impl(n, pvals, size=None):
            out = np.zeros((size, len(pvals)), res_dtype)
            common_impl_multinomial(n, pvals, out, dpnp_func, PRINT_DEBUG)
            return out

    elif isinstance(size, types.BaseTuple):

        def dpnp_impl(n, pvals, size=None):
            out = np.zeros(size + (len(pvals),), res_dtype)
            common_impl_multinomial(n, pvals, out, dpnp_func, PRINT_DEBUG)
            return out

    else:
        raise TypeError(
            "np.random.multinomial(): size should be int or "
            "tuple or None, got %s" % (size,)
        )

    return dpnp_impl


@overload(stubs.numpy.multivariate_normal)
def dpnp_random_impl(mean, cov, size=None, check_valid="warn", tol=1e-8):
    name = "multivariate_normal"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L239

    Function declaration:
    void custom_rng_multivariate_normal_c(void* result,
                                      const int dimen,
                                      const double* mean_vector,
                                      const size_t mean_vector_size,
                                      const double* cov_vector,
                                      const size_t cov_vector_size,
                                      size_t size)
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.int32,
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
        types.intp,
    )
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):

        def dpnp_impl(mean, cov, size=None, check_valid="warn", tol=1e-8):
            out = np.empty(mean.shape, dtype=res_dtype)
            common_impl_multivariate_normal(
                mean, cov, size, check_valid, tol, out, dpnp_func, PRINT_DEBUG
            )
            return out

    elif isinstance(size, types.Integer):

        def dpnp_impl(mean, cov, size=None, check_valid="warn", tol=1e-8):
            new_size = (size,)
            new_size = new_size + (mean.size,)
            out = np.empty(new_size, dtype=res_dtype)
            common_impl_multivariate_normal(
                mean, cov, size, check_valid, tol, out, dpnp_func, PRINT_DEBUG
            )
            return out

    elif isinstance(size, types.BaseTuple):

        def dpnp_impl(mean, cov, size=None, check_valid="warn", tol=1e-8):
            new_size = size + (mean.size,)
            out = np.empty(new_size, dtype=res_dtype)
            common_impl_multivariate_normal(
                mean, cov, size, check_valid, tol, out, dpnp_func, PRINT_DEBUG
            )
            return out

    else:
        raise TypeError(
            "np.random.multivariate_normal(): size should be int or "
            "tuple or None, got %s" % (size,)
        )

    return dpnp_impl


@overload(stubs.numpy.negative_binomial)
def dpnp_random_impl(n, p, size=None):
    name = "negative_binomial"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L267

    Function declaration:
    void custom_rng_negative_binomial_c(void* result, double a, double p, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.int32, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)
    res_dtype = np.int32
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(n, types.Integer)):
        raise ValueError("We only support scalar for input: n")

    if not (isinstance(p, types.Float)):
        raise ValueError("We only support scalar for input: p")

    if size in (None, types.none):

        def dpnp_impl(n, p, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(n, p, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(n, p, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(n, p, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.normal)
def dpnp_random_impl(loc=0.0, scale=1.0, size=None):
    name = "normal"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L282

    Function declaration:
    void custom_rng_normal_c(void* result, _DataType mean, _DataType stddev, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(loc, float):
        if not (isinstance(loc, types.Float)):
            raise ValueError("We only support scalar for input: loc")

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(loc=0.0, scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_2_arg(loc, scale, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.poisson)
def dpnp_random_impl(lam=1.0, size=None):
    name = "poisson"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L297

    Function declaration:
    void custom_rng_poisson_c(void* result, double lambda, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["int32", "NONE"], sig)
    res_dtype = np.int32
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(lam, float):
        if not (isinstance(lam, types.Float)):
            raise ValueError("We only support scalar for input: lam")

    if size in (None, types.none):

        def dpnp_impl(lam=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(lam, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(lam=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(lam, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.rayleigh)
def dpnp_random_impl(scale=1.0, size=None):
    name = "rayleigh"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L312

    Function declaration:
    void custom_rng_rayleigh_c(void* result, _DataType scale, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not isinstance(scale, float):
        if not (isinstance(scale, types.Float)):
            raise ValueError("We only support scalar for input: scale")

    if size in (None, types.none):

        def dpnp_impl(scale=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(scale, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(scale=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(scale, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.standard_cauchy)
def dpnp_random_impl(size=None):
    name = "standard_cauchy"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L331

    Function declaration:
    void custom_rng_standard_cauchy_c(void* result, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):

        def dpnp_impl(size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_0_arg(res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_0_arg(res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.standard_exponential)
def dpnp_random_impl(size=None):
    name = "standard_exponential"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L350

    Function declaration:
    void custom_rng_standard_exponential_c(void* result, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):

        def dpnp_impl(size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_0_arg(res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_0_arg(res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.standard_gamma)
def dpnp_random_impl(shape, size=None):
    name = "standard_gamma"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L364

    Function declaration:
    void custom_rng_standard_gamma_c(void* result, _DataType shape, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(shape, types.Float)):
        raise ValueError("We only support scalar for input: shape")

    if size in (None, types.none):

        def dpnp_impl(shape, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(shape, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(shape, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(shape, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.standard_normal)
def dpnp_random_impl(size=None):
    name = "normal"
    dpnp_lowering.ensure_dpnp("standard_normal")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L282

    Function declaration:
    void custom_rng_normal_c(void* result, _DataType mean, _DataType stddev, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):

        def dpnp_impl(size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(0.0, 1.0, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl(0.0, 1.0, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.uniform)
def dpnp_random_impl(low=0.0, high=1.0, size=None):
    name = "uniform"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L391

    Function declaration:
    void custom_rng_uniform_c(void* result, long low, long high, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.int64, types.int64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)

    res_dtype = np.float64

    PRINT_DEBUG = dpnp_lowering.DEBUG

    if size in (None, types.none):

        def dpnp_impl(low=0.0, high=1.0, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl(low, high, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(low=0.0, high=1.0, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl(low, high, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl


@overload(stubs.numpy.weibull)
def dpnp_random_impl(a, size=None):
    name = "weibull"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_random.cpp#L411

    Function declaration:
    void custom_rng_weibull_c(void* result, double alpha, size_t size)
    """
    sig = signature(ret_type, types.voidptr, types.float64, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, ["float64", "NONE"], sig)
    res_dtype = np.float64
    PRINT_DEBUG = dpnp_lowering.DEBUG

    if not (isinstance(a, types.Float)):
        raise ValueError("We only support scalar for input: a")

    if size in (None, types.none):

        def dpnp_impl(a, size=None):
            res = np.empty(1, dtype=res_dtype)
            common_impl_1_arg(a, res, dpnp_func, PRINT_DEBUG)
            return res[0]

    else:

        def dpnp_impl(a, size=None):
            res = np.empty(size, dtype=res_dtype)
            if res.size != 0:
                common_impl_1_arg(a, res, dpnp_func, PRINT_DEBUG)
            return res

    return dpnp_impl
