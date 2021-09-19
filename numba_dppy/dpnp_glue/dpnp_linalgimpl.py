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

import numba_dppy
import numba_dppy.dpnp_glue as dpnp_lowering
import numba_dppy.dpnp_glue.dpnpimpl as dpnp_ext
from numba_dppy import dpctl_functions

from . import stubs


@overload(stubs.dpnp.eig)
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
    PRINT_DEBUG = dpnp_lowering.DEBUG

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
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        wr_usm = dpctl_functions.malloc_shared(wr.size * wr.itemsize, sycl_queue)
        vr_usm = dpctl_functions.malloc_shared(vr.size * vr.itemsize, sycl_queue)

        dpnp_eig(a_usm, wr_usm, vr_usm, n)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, wr.ctypes, wr_usm, wr.size * wr.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, vr.ctypes, vr_usm, vr.size * vr.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(wr_usm, sycl_queue)
        dpctl_functions.free_with_queue(vr_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([wr.size, vr.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return (wr, vr)

    return dpnp_impl


@register_jitable
def common_matmul_impl(dpnp_func, a, b, out, m, n, k, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()

    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    dpctl_functions.queue_memcpy(sycl_queue, a_usm, a.ctypes, a.size * a.itemsize)

    b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, b_usm, b.ctypes, b.size * b.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    a_m = np.array((m, k))
    b_m = np.array((k, n))
    out_m = np.array((m, n))

    result_out = out_usm
    result_size = out.size
    result_ndim = 2
    result_shape = out_m.ctypes
    result_strides = 0

    input1_in = a_usm
    input1_size = a.size
    input1_ndim = 2
    input1_shape = a_m.ctypes
    input1_strides = 0

    input2_in = b_usm
    input2_size = b.size
    input2_ndim = 2
    input2_shape = b_m.ctypes
    input2_strides = 0

    dpnp_func(
        result_out,
        result_size,
        result_ndim,
        result_shape,
        result_strides,
        input1_in,
        input1_size,
        input1_ndim,
        input1_shape,
        input1_strides,
        input2_in,
        input2_size,
        input2_ndim,
        input2_shape,
        input2_strides,
    )

    event = dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    dpctl_functions.free_with_queue(a_usm, sycl_queue)
    dpctl_functions.free_with_queue(b_usm, sycl_queue)
    dpctl_functions.free_with_queue(out_usm, sycl_queue)

    dpnp_ext._dummy_liveness_func([a.size, b.size, out.size])
    dpnp_ext._dummy_liveness_func([a_m.size, b_m.size, out_m.size])

    if print_debug:
        print("dpnp implementation")


@register_jitable
def common_dot_impl(dpnp_func, a, b, out, m, print_debug):
    sycl_queue = dpctl_functions.get_current_queue()
    a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    b_usm = dpctl_functions.malloc_shared(b.size * b.itemsize, sycl_queue)
    event = dpctl_functions.queue_memcpy(
        sycl_queue, b_usm, b.ctypes, b.size * b.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

    out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

    result_out = out_usm
    result_size = out.size
    result_ndim = out.ndim
    result_shape = out.shapeptr
    result_strides = 0

    input1_in = a_usm
    input1_size = a.size
    input1_ndim = a.ndim
    input1_shape = a.shapeptr
    input1_strides = 0

    input2_in = b_usm
    input2_size = b.size
    input2_ndim = b.ndim
    input2_shape = b.shapeptr
    input2_strides = 0

    dpnp_func(
        result_out,
        result_size,
        result_ndim,
        result_shape,
        result_strides,
        input1_in,
        input1_size,
        input1_ndim,
        input1_shape,
        input1_strides,
        input2_in,
        input2_size,
        input2_ndim,
        input2_shape,
        input2_strides,
    )

    event = dpctl_functions.queue_memcpy(
        sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
    )
    dpctl_functions.event_wait(event)
    dpctl_functions.event_delete(event)

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


@overload(stubs.dpnp.matmul)
@overload(stubs.dpnp.dot)
def dpnp_dot_impl(a, b):
    dpnp_lowering.ensure_dpnp("dot")

    ret_type = types.void
    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blame/67a101c90cf253cfe9b9ba80ac397811ce94edee/dpnp/backend/kernels/dpnp_krnl_common.cpp#L322

    Function declaration:
    void dpnp_matmul_c(void* result_out,
                    const size_t result_size,
                    const size_t result_ndim,
                    const size_t* result_shape,
                    const size_t* result_strides,
                    const void* input1_in,
                    const size_t input1_size,
                    const size_t input1_ndim,
                    const size_t* input1_shape,
                    const size_t* input1_strides,
                    const void* input2_in,
                    const size_t input2_size,
                    const size_t input2_ndim,
                    const size_t* input2_shape,
                    const size_t* input2_strides)
    """
    sig = signature(
        ret_type,
        types.voidptr,
        types.intp,
        types.intp,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.intp,
        types.voidptr,
        types.voidptr,
        types.voidptr,
        types.intp,
        types.intp,
        types.voidptr,
        types.voidptr,
    )

    res_dtype = get_res_dtype(a, b)

    PRINT_DEBUG = dpnp_lowering.DEBUG

    ndims = [a.ndim, b.ndim]
    if ndims == [2, 2]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dpnp_impl_dot_2_mm(a, b):
            m, k = a.shape
            _k, n = b.shape

            if _k != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((m, n), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dpnp_impl_dot_2_mm
    elif ndims == [2, 1]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dpnp_impl_dot_2_mv(a, b):
            m, k = a.shape
            (_n,) = b.shape
            n = 1

            if _n != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((m,), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dpnp_impl_dot_2_mv
    elif ndims == [1, 2]:
        dpnp_func = dpnp_ext.dpnp_func("dpnp_matmul", [a.dtype.name, "NONE"], sig)

        def dpnp_impl_dot_2_vm(a, b):
            (m,) = a.shape
            k, n = b.shape

            if m != k:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty((n,), dtype=res_dtype)
            common_matmul_impl(dpnp_func, a, b, out, m, n, k, PRINT_DEBUG)

            return out

        return dpnp_impl_dot_2_vm
    elif ndims == [1, 1]:
        """
        dpnp source:
        https://github.com/IntelPython/dpnp/blob/67a101c90cf253cfe9b9ba80ac397811ce94edee/dpnp/backend/kernels/dpnp_krnl_common.cpp#L79

        Function declaration:
        void dpnp_dot_c(void* result_out,
                        const size_t result_size,
                        const size_t result_ndim,
                        const size_t* result_shape,
                        const size_t* result_strides,
                        const void* input1_in,
                        const size_t input1_size,
                        const size_t input1_ndim,
                        const size_t* input1_shape,
                        const size_t* input1_strides,
                        const void* input2_in,
                        const size_t input2_size,
                        const size_t input2_ndim,
                        const size_t* input2_shape,
                        const size_t* input2_strides)
        """
        sig = signature(
            ret_type,
            types.voidptr,
            types.intp,
            types.intp,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.intp,
            types.intp,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.intp,
            types.intp,
            types.voidptr,
            types.voidptr,
        )
        dpnp_func = dpnp_ext.dpnp_func("dpnp_dot", [a.dtype.name, "NONE"], sig)

        def dpnp_impl_dot_2_vv(a, b):

            (m,) = a.shape
            (n,) = b.shape

            if m != n:
                raise ValueError("Incompatible array sizes for np.dot(a, b)")

            out = np.empty(1, dtype=res_dtype)
            common_dot_impl(dpnp_func, a, b, out, m, PRINT_DEBUG)

            return out[0]

        return dpnp_impl_dot_2_vv
    else:
        assert 0


@overload(stubs.dpnp.multi_dot)
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
            result = numba_dppy.dpnp.dot(result, arrays[idx])

        if print_debug:
            print("dpnp implementation")
        return result

    return dpnp_impl


@overload(stubs.dpnp.vdot)
def dpnp_vdot_impl(a, b):
    dpnp_lowering.ensure_dpnp("vdot")

    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L118

    Function declaration:
    void dpnp_dot_c(void* array1_in, void* array2_in, void* result1, size_t size)

    """

    def dpnp_impl(a, b):
        return numba_dppy.dpnp.dot(np.ravel(a), np.ravel(b))

    return dpnp_impl


@overload(stubs.dpnp.matrix_power)
def dpnp_matrix_power_impl(a, n):
    dpnp_lowering.ensure_dpnp("matrix_power")

    """
    dpnp source:
    https://github.com/IntelPython/dpnp/blob/0.4.0/dpnp/backend/custom_kernels.cpp#L42

    Function declaration:
    void dpnp_matmul_c(void* array1_in, void* array2_in, void* result1, size_t size_m,
                       size_t size_n, size_t size_k)
    """

    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a, n):
        if n < 0:
            raise ValueError("n < 0 is not supported for np.linalg.matrix_power(a, n)")

        if n == 0:
            if PRINT_DEBUG:
                print("dpnp implementation")
            return np.identity(a.shape[0], a.dtype)

        result = a
        for idx in range(0, n - 1):
            result = numba_dppy.dpnp.matmul(result, a)
        return result

    return dpnp_impl


@overload(stubs.dpnp.cholesky)
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
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(a):
        n = a.shape[-1]
        if a.shape[-2] != n:
            raise ValueError("Input array must be square.")

        out = a.copy()

        if n == 0:
            return out

        sycl_queue = dpctl_functions.get_current_queue()
        a_usm = dpctl_functions.malloc_shared(a.size * a.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, a.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return out

    return dpnp_impl


@overload(stubs.dpnp.det)
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
    PRINT_DEBUG = dpnp_lowering.DEBUG

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
        event = dpctl_functions.queue_memcpy(
            sycl_queue, a_usm, a.ctypes, a.size * a.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(a_usm, out_usm, a.shapeptr, a.ndim)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(a_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, a.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        if a.ndim == 2:
            return out[0]
        else:
            return out

    return dpnp_impl


@overload(stubs.dpnp.matrix_rank)
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
    PRINT_DEBUG = dpnp_lowering.DEBUG

    def dpnp_impl(M, tol=None, hermitian=False):
        if tol is not None:
            raise ValueError("tol is not supported for np.linalg.matrix_rank(M)")
        if hermitian:
            raise ValueError("hermitian is not supported for np.linalg.matrix_rank(M)")

        if M.ndim > 2:
            raise ValueError(
                "np.linalg.matrix_rank(M) is only supported on 1 or 2-d arrays"
            )

        out = np.empty(1, dtype=M.dtype)

        sycl_queue = dpctl_functions.get_current_queue()
        M_usm = dpctl_functions.malloc_shared(M.size * M.itemsize, sycl_queue)
        event = dpctl_functions.queue_memcpy(
            sycl_queue, M_usm, M.ctypes, M.size * M.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        out_usm = dpctl_functions.malloc_shared(out.size * out.itemsize, sycl_queue)

        dpnp_func(M_usm, out_usm, M.shapeptr, M.ndim)

        event = dpctl_functions.queue_memcpy(
            sycl_queue, out.ctypes, out_usm, out.size * out.itemsize
        )
        dpctl_functions.event_wait(event)
        dpctl_functions.event_delete(event)

        dpctl_functions.free_with_queue(M_usm, sycl_queue)
        dpctl_functions.free_with_queue(out_usm, sycl_queue)

        dpnp_ext._dummy_liveness_func([out.size, M.size])

        if PRINT_DEBUG:
            print("dpnp implementation")
        return out[0]

    return dpnp_impl


@overload(stubs.dpnp.eigvals)
def dpnp_eigvals_impl(a):
    dpnp_lowering.ensure_dpnp("eigvals")

    def dpnp_impl(a):
        eigval, eigvec = numba_dppy.dpnp.eig(a)
        return eigval

    return dpnp_impl
