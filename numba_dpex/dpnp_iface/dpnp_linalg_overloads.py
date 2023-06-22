# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import dpnp
from llvmlite import ir as llvmir
from numba import types
from numba.core import cgutils

# from numba.core.typing import signature
from numba.extending import intrinsic, overload
from numba.np.arrayobj import _parse_shape, make_array

import numba_dpex.onemkl
import numba_dpex.utils as utils
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.dpnp_iface._intrinsic import (
    _empty_nd_impl,
    _get_queue_ref,
    alloc_empty_arrayobj,
    fill_arrayobj,
)

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)


def _parse_dtypes(a):
    # if dpnp.issubdtype(a_dtype, dpnp.complexfloating):
    if "complex" in str(a.dtype):
        v_type = a.dtype
        w_type = dpnp.float64 if a.dtype == dpnp.complex128 else dpnp.float32
    # elif dpnp.issubdtype(a_dtype, dpnp.floating):
    elif "float" in str(a.dtype):
        v_type = w_type = a.dtype
    elif a.queue.sycl_device.has_aspect_fp64:
        v_type = w_type = dpnp.float64
    else:
        v_type = w_type = dpnp.float32
    return (v_type, w_type)


def _parse_lapack_func(a):
    if "complex" in str(a.dtype):
        return "_heevd"
    else:
        return "_syevd"


@intrinsic
def impl_dpnp_linalg_eigh(
    ty_context, ty_a, ty_v, ty_w, ty_lda, ty_n, ty_uplo, ty_sycl_queue
):
    ty_retty_ = types.none
    signature = ty_retty_(
        ty_a, ty_v, ty_w, ty_lda, ty_n, ty_uplo, ty_sycl_queue
    )

    def codegen(context, builder, sig, args):
        mod = builder.module

        qref_payload: _QueueRefPayload = _get_queue_ref(
            context, builder, args[-1], sig.args[-1], sig.args[-1].instance_type
        )

        lda = context.get_argument_value(builder, sig.args[3], args[3])
        n = context.get_argument_value(builder, sig.args[4], args[4])
        uplo = context.get_argument_value(builder, sig.args[5], args[5])
        _lda = builder.bitcast(lda, llvmir.IntType(64))
        _n = builder.bitcast(n, llvmir.IntType(64))
        _uplo = builder.bitcast(uplo, llvmir.IntType(64))

        with builder.goto_entry_block():
            a_ptr = cgutils.alloca_once(builder, args[0].type)
            v_ptr = cgutils.alloca_once(builder, args[1].type)
            w_ptr = cgutils.alloca_once(builder, args[2].type)

        builder.store(args[0], a_ptr)
        builder.store(args[1], v_ptr)
        builder.store(args[2], w_ptr)

        _a_ptr = builder.bitcast(a_ptr, cgutils.voidptr_t)
        _v_ptr = builder.bitcast(v_ptr, cgutils.voidptr_t)
        _w_ptr = builder.bitcast(w_ptr, cgutils.voidptr_t)

        u64 = llvmir.IntType(64)
        fnty = llvmir.FunctionType(
            utils.LLVMTypes.void_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                u64,
                u64,
                u64,
            ],
        )

        fn = cgutils.get_or_insert_function(
            mod, fnty, "DPEX_ONEMKL_LAPACK_syevd"
        )
        builder.call(fn, [_a_ptr, _v_ptr, _w_ptr, _lda, _n, _uplo])

        if qref_payload.py_dpctl_sycl_queue_addr:
            qref_payload.pyapi.decref(qref_payload.py_dpctl_sycl_queue_addr)

        return None

    return signature, codegen


"""
1. type_w (vector), type_v(matrix) using DpnpNdArray type before impl (overload typing section)
    ndim = 1 for vector, ndim = 2 matrix
2. impl(a, UPLO):
    if a.ndim > 2:
        not implemented error
    else:
        i. impl_eigh(ty_a, ty_UPLO, type_w, type_v):
            def codegen(ctx, bldr, sig, args):
                ii. getting the shape out of a, _parse_shapes(args[0], ...)
                iii. call something like empty_nd_impl, lapack_eigh_wrapper(d1,d2)
                    v, q = empty_nd_impl()
                    w, q = empty_nd_impl()
                    call cpp function (a, w, v, q, UPLO)
"""


@overload(dpnp.linalg.eigh, prefer_literal=True)
def ol_dpnp_linalg_eigh(a, UPLO="L"):
    _upper_lower = {"U": 0, "L": 1}

    _usm_type = a.usm_type
    _sycl_queue = a.queue
    _order = "C" if a.is_c_contig else "F"
    _uplo = _upper_lower[UPLO]

    _lapack_func = _parse_lapack_func(a)  # noqa: F841
    _v_type, _w_type = _parse_dtypes(a)

    def impl(
        a,
        UPLO="L",
    ):
        # if a.ndim > 2:
        #     pass
        # else:
        #     # oneMKL LAPACK assumes fortran-like array as input, so
        #     # allocate a memory with 'F' order for dpnp array of eigenvectors
        #     v = dpnp.empty_like(a, order="F", dtype=v_type)

        #     # # use DPCTL tensor function to fill the array of eigenvectors with content of input array
        #     # ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(src=a_usm_arr, dst=v.get_array(), sycl_queue=a_sycl_queue)

        #     # # allocate a memory for dpnp array of eigenvalues
        #     w = dpnp.empty(
        #         a.shape[:-1], dtype=w_type, usm_type=a.usm_type
        #     )  # sycl_queue=a.queue)

        #     # insert_lapack_eigh(a)
        #     insert_lapack_eigh()

        #     # # call LAPACK extension function to get eigenvalues and eigenvectors of matrix A
        #     # ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(a_sycl_queue, jobz, uplo, v.get_array(), w.get_array(), depends=[copy_ev])

        #     if a.order != "F":
        #         # need to align order of eigenvectors with one of input matrix A
        #         out_v = dpnp.empty_like(v, order=a.order)
        #         # ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(src=v.get_array(), dst=out_v.get_array(), sycl_queue=a_sycl_queue, depends=[lapack_ev])
        #         # ht_copy_out_ev.wait()
        #     else:
        #         out_v = v

        #     # ht_lapack_ev.wait()
        #     # ht_copy_ev.wait()

        #     # w = dpnp.ones((3,3))

        # return (w, out_v)

        lda, n = a.shape
        if lda != n:
            raise ValueError("Last 2 dimensions of the array must be square.")

        v = dpnp.empty(
            (lda, n), dtype=_v_type, order=_order, usm_type=_usm_type
        )
        w = dpnp.empty((1, n), dtype=_w_type, order=_order, usm_type=_usm_type)

        impl_dpnp_linalg_eigh(a, v, w, lda, n, _uplo, _sycl_queue)
        return (w, v.T)

    return impl
