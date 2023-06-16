# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import dpnp
from llvmlite import ir as llvmir
from numba import types
from numba.core import cgutils
from numba.core.typing import signature
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
# def insert_lapack_eigh(context, a):
def insert_lapack_eigh(context):
    # v = dpnp.empty(10, sycl_queue=a.queue)  # noqa: F841

    def codegen(context, builder, sig, args):
        mod = builder.module
        # breakpoint()
        print("type(builder) =", type(builder))
        # with builder.goto_entry_block():
        #     ptr = cgutils.alloca_once(builder, args[0].type)
        # builder.store(args[0], ptr)
        # _ptr = builder.bitcast(ptr, cgutils.voidptr_t)
        # fnty = llvmir.FunctionType(utils.LLVMTypes.void_t, [cgutils.voidptr_t])
        fnty = llvmir.FunctionType(utils.LLVMTypes.void_t, [])
        fn = cgutils.get_or_insert_function(mod, fnty, "DPEX_LAPACK_eigh")
        # ret = builder.call(fn, [_ptr])  # noqa: F841
        ret = builder.call(fn, [])  # noqa: F841
        return ret

    # sig = signature(types.void, a)
    sig = signature(types.void, types.void)
    return sig, codegen


@intrinsic
def impl_dpnp_lapack_eigh(ty_context, ty_a, ty_UPLO, ty_sycl_queue, ty_w):
    # print("ty_retty_ref =", ty_retty_ref)
    # print("type(ty_retty_ref) =", type(ty_retty_ref))
    # print("ty_retty_ref.instance_type =", ty_retty_ref.instance_type)
    # print("type(ty_retty_ref.instance_type) =", type(ty_retty_ref.instance_type))
    ty_retty = ty_w.instance_type
    signature = ty_retty(ty_a, ty_UPLO, ty_sycl_queue, ty_w)

    def codegen(context, builder, sig, args):
        # print("ty_retty =", ty_retty)
        # print("type(ty_retty) =", type(ty_retty))
        # breakpoint()
        qref_payload: _QueueRefPayload = _get_queue_ref(  # noqa: F841
            context, builder, args[-2], sig.args[-2], sig.return_type.queue
        )

        print("qref_payload.queue_ref =", qref_payload.queue_ref)
        ary = alloc_empty_arrayobj(
            context, builder, sig, qref_payload.queue_ref, args, is_like=True
        )

        fill_value = context.get_constant(types.intp, 1)
        ary, _ = fill_arrayobj(
            context,
            builder,
            ary,
            sig.return_type,
            qref_payload.queue_ref,
            fill_value,
        )

        if qref_payload.py_dpctl_sycl_queue_addr:
            qref_payload.pyapi.decref(qref_payload.py_dpctl_sycl_queue_addr)

        return ary._getvalue()
        # mod = builder.module  # noqa: F841
        # fnty = llvmir.FunctionType(utils.LLVMTypes.void_t, [])
        # fn = cgutils.get_or_insert_function(mod, fnty, "DPEX_LAPACK_eigh")
        # ret = builder.call(fn, [])  # noqa: F841
        # return ret
        # ary = alloc_empty_arrayobj(
        #     context, builder, sig, qref_payload.queue_ref, args, is_like=True
        # )
        # fill_value = context.get_argument_value(builder, sig.args[1], args[1])
        # ary, _ = fill_arrayobj(
        #     context,
        #     builder,
        #     ary,
        #     sig.return_type,
        #     qref_payload.queue_ref,
        #     fill_value,
        # )
        # if qref_payload.py_dpctl_sycl_queue_addr:
        #     qref_payload.pyapi.decref(qref_payload.py_dpctl_sycl_queue_addr)

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
    # _jobz = {"N": 0, "V": 1}
    _upper_lower = {"U": 0, "L": 1}

    print("a =", a, "type(a) =", type(a))
    _ndim = a.ndim
    _usm_type = a.usm_type
    _sycl_queue = a.queue
    print("_sycl_queue =", _sycl_queue)
    # a_order = "C" if a.is_c_contig else "F"
    _layout = "C" if a.is_c_contig else "F"  # noqa: F841
    # a_usm_arr = dpnp.get_usm_ndarray(a)

    # 'V' means both eigenvectors and eigenvalues will be calculated
    # jobz = _jobz["V"]  # noqa: F841
    jobz = 1  # noqa: F841
    uplo = _upper_lower[UPLO]  # noqa: F841

    # get resulting type of arrays with eigenvalues and eigenvectors
    # a_dtype = a.dtype
    # lapack_func = "_syevd"  # noqa: F841
    lapack_func = _parse_lapack_func(a)  # noqa: F841
    v_type, w_type = _parse_dtypes(a)  # noqa: F841

    type_w = DpnpNdArray(
        ndim=1,
        layout=_layout,
        dtype=w_type,
        usm_type=_usm_type,
        # device=a.queue.sycl_device,
        queue=_sycl_queue,
    )

    type_v = DpnpNdArray(
        ndim=_ndim,
        layout=_layout,
        dtype=v_type,
        usm_type=_usm_type,
        # device=a.queue.sycl_device,
        queue=_sycl_queue,
    )

    print("type_w =", type_w)
    print("type(type_w) =", type(type_w))
    print("type_v =", type_v)
    print("type(type_v) =", type_v)

    ret_ty = types.Tuple((type_w, type_v))  # noqa: F841

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
        return impl_dpnp_lapack_eigh(a, UPLO, _sycl_queue, type_w)

    return impl


# from numba.np import numpy_support as np_support

# @overload(dpnp.linalg.eigh)
# def eigh_impl(a):
#     # ensure_lapack()

#     # _check_linalg_matrix(a, "eigh")

#     # convert typing floats to numpy floats for use in the impl
#     w_type = getattr(a.dtype, "underlying_float", a.dtype)
#     w_dtype = np_support.as_dtype(w_type)

#     # numba_ez_xxxevd = _LAPACK().numba_ez_xxxevd(a.dtype)

#     # kind = ord(get_blas_kind(a.dtype, "eigh"))

#     JOBZ = ord('V')
#     UPLO = ord('L')

#     def eigh_impl(a):
#         n = a.shape[-1]

#         # if a.shape[-2] != n:
#         #     msg = "Last 2 dimensions of the array must be square."
#         #     raise np.linalg.LinAlgError(msg)

#         # _check_finite_matrix(a)

#         acpy = a # _copy_to_fortran_order(a) # write an intrinsic

#         w = dpnp.ones((3,3), dtype=w_dtype)
#         insert_lapack_eigh(w)

#         # if n == 0:
#         #     return (w, acpy)

#         # r = numba_ez_xxxevd(kind,  # kind
#         #                     JOBZ,  # jobz
#         #                     UPLO,  # uplo
#         #                     n,  # n
#         #                     acpy.ctypes,  # a
#         #                     n,  # lda
#         #                     w.ctypes  # w
#         #                     )
#         # _handle_err_maybe_convergence_problem(r)

#         # help liveness analysis
#         # _dummy_liveness_func([acpy.size, w.size])
#         return (w, acpy)

#     return eigh_impl
