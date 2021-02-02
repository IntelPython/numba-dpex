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
import numba_dppy


@overload(stubs.numpy.eig)
def dpnp_eig_impl(a):
    name = "eig"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L180

    Function declaration:
    void dpnp_eig_c(const void* array_in, void* result1, void* result2, size_t size)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp)
    dpnp_eig = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    res_dtype = np.float64
    if a.dtype == types.float32:
        res_dtype = np.float32

    def dpnp_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            msg = "Last 2 dimensions of the array must be square."
            raise ValueError(msg)

        dpnp_ext._check_finite_matrix(a)

        wr = np.empty(n, dtype=res_dtype)
        vr = np.empty((n, n), dtype=res_dtype)

        if n == 0:
            return (wr, vr)

        sycl_queue = dpctl_functions.get_current_queue()
        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        wr_usm = dpctl_functions.malloc_shared(wr.size * wr.itemsize, sycl_queue)
        vr_usm = dpctl_functions.malloc_shared(vr.size * vr.itemsize, sycl_queue)

        dpnp_eig(a_usm, wr_usm, vr_usm, n)

        dpctl_functions.queue_memcpy(
            sycl_queue, wr.ctypes, wr_usm, wr.size * wr.itemsize
        )
        dpctl_functions.queue_memcpy(
            sycl_queue, vr.ctypes, vr_usm, vr.size * vr.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(wr_usm, sycl_queue)
        dpctl_functions.free_with_queue(vr_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([wr.size, vr.size])

        return (wr, vr)

    return dpnp_impl


@register_jitable
def common_matmul_impl(dpnp_func, a, b, out, m, n, k, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()

    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

    b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, b_usm, b.ctypes, b.size * b.itemsize)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    dpnp_func(a_usm, b_usm, out_usm, m, n, k)

    dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(b_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, b.size, out.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_dot_impl(dpnp_func, a, b, out, m, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

    b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, b_usm, b.ctypes, b.size * b.itemsize)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    dpnp_func(a_usm, b_usm, out_usm, m)

    dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(b_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, b.size, out.size])

    if print_debug:
        print("dpnp implementation")


def get_res_dtype(a, b):
    res_dtype = np.float64
    if a.dtype == types.int32 and b.dtype == types.int32:
        res_dtype = np.int32
    elif a.dtype == types.int32 and b.dtype == types.int64:
        res_dtype = np.int64
    elif a.dtype == types.int32 and b.dtype == types.float32:
        res_dtype = np.float64
    elif a.dtype == types.int32 and b.dtype == types.float64:
        res_dtype = np.float64
    elif a.dtype == types.int64 and b.dtype == types.int32:
        res_dtype = np.int64
    elif a.dtype == types.int64 and b.dtype == types.int64:
        res_dtype = np.int64
    elif a.dtype == types.int64 and b.dtype == types.float32:
        res_dtype = np.float64
    elif a.dtype == types.int64 and b.dtype == types.float64:
        res_dtype = np.float64
    elif a.dtype == types.float32 and b.dtype == types.int32:
        res_dtype = np.float64
    elif a.dtype == types.float32 and b.dtype == types.int64:
        res_dtype = np.float64
    elif a.dtype == types.float32 and b.dtype == types.float32:
        res_dtype = np.float32
    elif a.dtype == types.float32 and b.dtype == types.float64:
        res_dtype = np.float64
    elif a.dtype == types.float64 and b.dtype == types.int32:
        res_dtype = np.float64
    elif a.dtype == types.float64 and b.dtype == types.int64:
        res_dtype = np.float64
    elif a.dtype == types.float64 and b.dtype == types.float32:
        res_dtype = np.float32
    elif a.dtype == types.float64 and b.dtype == types.float64:
        res_dtype = np.float64

    return res_dtype


@overload(stubs.numpy.matmul)
@overload(stubs.numpy.dot)
def dpnp_dot_impl(a, b):
    dpnp_lowering.ensure_dpnp("dot")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L42
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L118

    Function declaration:
    void dpnp_matmul_c(void* array1_in, void* array2_in, void* result1, size_t size_m,
                       size_t size_n, size_t size_k)
    void dpnp_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)

    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.intp,
        types.intp,
    )

    res_dtype = get_res_dtype(a, b)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    ndims = [a.ndim, b.ndim]
    if ndims == [2, 2]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dot_2_mm(a, b):
            m, k = a.shape
            _k, n = b.shape

            if _k != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((m, n), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dot_2_mm
    elif ndims == [2, 1]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dot_2_mv(a, b):
            m, k = a.shape
            (_n,) = b.shape
            n = 1

            if _n != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((m,), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dot_2_mv
    elif ndims == [1, 2]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dot_2_vm(a, b):
            (m,) = a.shape
            k, n = b.shape

            if m != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((n,), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dot_2_vm
    elif ndims == [1, 1]:
        sig = signature(
            ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp
        )
        dpnp_func = dpnp_ext.dpnp_func("dpnp_dot", [a.dtype.name, "NONE"], sig)

        def dot_2_vv(a, b):

            (m,) = a.shape
            (n,) = b.shape

            if m != n:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty(1, dtype=res_dtype)
            common_dot_impl(dpnp_func, a, b, out, m, PRINT_DEBUG)

            return out[0]

        return dot_2_vv
    else:
        assert 0


@overload(stubs.numpy.multi_dot)
def dpnp_multi_dot_impl(arrays):
    dpnp_lowering.ensure_dpnp("multi_dot")

    print_debug = dpnp_lowering.DEBUG
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L118

    Function declaration:
    void dpnp_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)

    """

    def dpnp_impl(arrays):
        n = len(arrays)
        result = arrays[0]

        for idx in range(1, n):
            result = numba_dppy.numpy.dot(result, arrays[idx])

        if print_debug:
            print("dpnp implementation")
        return result

    return dpnp_impl


@overload(stubs.numpy.vdot)
def dpnp_vdot_impl(a, b):
    dpnp_lowering.ensure_dpnp("vdot")

    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L118

    Function declaration:
    void dpnp_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)

    """

    def dpnp_impl(a, b):
        return numba_dppy.numpy.dot(np.ravel(a), np.ravel(b))

    return dpnp_impl


@overload(stubs.numpy.matrix_power)
def dpnp_matrix_power_impl(a, n):
    dpnp_lowering.ensure_dpnp("matrix_power")

    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L42

    Function declaration:
    void dpnp_matmul_c(void* array1_in, void* array2_in, void* result1, size_t size_m,
                       size_t size_n, size_t size_k)
    """

    def dpnp_impl(a, n):
        if n < 0:
            raise ValueError("n < 0 is not supported for np.linalg.matrix_power(a, n)")

        if n == 0:
            return np.identity(a.shape[0], a.dtype)

        result = a
        for idx in range(0, n - 1):
            result = numba_dppy.numpy.matmul(result, a)
        return result

    return dpnp_impl


@overload(stubs.numpy.cholesky)
def dpnp_cholesky_impl(a):
    name = "cholesky"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_linalg.cpp#L40

    Function declaration:
    void custom_cholesky_c(void* array1_in, void* result1, size_t* shape)

    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.voidptr)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    def dpnp_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            raise ValueError("Input array must be square.")

        out = a.copy()

        if n == 0:
            return out

        sycl_queue = dpctl_functions.get_current_queue()
        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, a.size])

        return out

    return dpnp_impl


@overload(stubs.numpy.det)
def dpnp_det_impl(a):
    name = "det"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_linalg.cpp#L83

    Function declaration:
    void custom_det_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [a.dtype.name, "NONE"], sig)

    def dpnp_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            raise ValueError("Input array must be square.")

        dpnp_ext._check_finite_matrix(a)

        if a.ndim == 2:
            out = np.empty((1,), dtype=a.dtype)
            out[0] = -4
        else:
            out = np.empty(a.shape[:-2], dtype=a.dtype)

        sycl_queue = dpctl_functions.get_current_queue()
        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, a.size])

        if a.ndim == 2:
            return out[0]
        else:
            return out

    return dpnp_impl


@overload(stubs.numpy.matrix_rank)
def dpnp_matrix_rank_impl(M, tol=None, hermitian=False):
    name = "matrix_rank"
    dpnp_lowering.ensure_dpnp(name)

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels_linalg.cpp#L186

    Function declaration:
    void custom_matrix_rank_c(void* array1_in, void* result1, size_t* shape, size_t ndim)
    """
    sig = signature(ret_type, types.voidptr, types.voidptr, types.voidptr, types.intp)
    dpnp_func = dpnp_ext.dpnp_func("dpnp_" + name, [M.dtype.name, "NONE"], sig)

    def dpnp_impl(M, tol=None, hermitian=False):
        if tol != None:
            raise ValueError("tol is not supported for np.linalg.matrix_rank(M)")
        if hermitian == True:
            raise ValueError("hermitian is not supported for np.linalg.matrix_rank(M)")

        if M.ndim > 2:
            raise ValueError(
                "np.linalg.matrix_rank(M) is only supported on 1 or 2-d arrays"
            )

        out = np.empty(1, dtype=M.dtype)

        sycl_queue = dpctl_functions.get_current_queue()
        M_usm = dpctl_functions.malloc_shared(M.size * M.itemsize, sycl_queue)
        dpctl_functions.queue_memcpy(sycl_queue, M_usm, M.ctypes, M.size * M.itemsize)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(M_usm, out_usm, M.shapeptr, M.ndim)

        dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )

        dpctl_functions.free_with_queue(M_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, M.size])

        return out[0]

    return dpnp_impl


@overload(stubs.numpy.eigvals)
def dpnp_eigvals_impl(a):
    dpnp_lowering.ensure_dpnp("eigvals")

    def dpnp_impl(a):
        eigval, eigvec = numba_dppy.numpy.eig(a)
        return eigval

    return dpnp_impl
