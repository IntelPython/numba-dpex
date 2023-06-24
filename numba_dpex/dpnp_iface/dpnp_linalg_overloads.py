# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import dpnp
import numpy as np
from llvmlite import ir as llvmir
from numba import types
from numba.core import cgutils
from numba.core.types.scalars import Complex, Float, Integer
from numba.extending import intrinsic, overload

import numba_dpex.onemkl
import numba_dpex.utils as utils
from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.dpnp_iface._intrinsic import _get_queue_ref

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)


def _parse_dtypes(a):
    if isinstance(a.dtype, Complex):
        v_type = a.dtype
        w_type = dpnp.float64 if a.dtype.bitwidth == 128 else dpnp.float32
    elif isinstance(a.dtype, Float):
        v_type = w_type = a.dtype
    elif isinstance(a.dtype, Integer):
        v_type = w_type = (
            dpnp.float32 if a.dtype.bitwidth == 32 else dpnp.float64
        )
    # elif a.queue.sycl_device.has_aspect_fp64:
    #     v_type = w_type = dpnp.float64
    else:
        v_type = w_type = dpnp.float64
    return (v_type, w_type)


@intrinsic
def impl_dpnp_linalg_eigh(
    ty_context, ty_a, ty_v, ty_w, ty_lda, ty_n, ty_uplo, ty_sycl_queue
):
    lapack_func_name = "DPEX_ONEMKL_LAPACK_syevd"
    if isinstance(ty_a.dtype, Complex):
        lapack_func_name = "DPEX_ONEMKL_LAPACK_heevd"

    ty_retty_ = types.none
    signature = ty_retty_(
        ty_a, ty_v, ty_w, ty_lda, ty_n, ty_uplo, ty_sycl_queue
    )

    def codegen(context, builder, sig, args):
        u64 = llvmir.IntType(64)
        mod = builder.module

        qref_payload: _QueueRefPayload = _get_queue_ref(
            context, builder, args[-1], sig.args[-1], sig.args[-1].instance_type
        )

        dpexrtCtx = dpexrt.DpexRTContext(context)
        _queue_ref = dpexrtCtx.copy_queue(builder, qref_payload.queue_ref)

        lda = context.get_argument_value(builder, sig.args[3], args[3])
        n = context.get_argument_value(builder, sig.args[4], args[4])
        uplo = context.get_argument_value(builder, sig.args[5], args[5])
        _lda = builder.bitcast(lda, u64)
        _n = builder.bitcast(n, u64)
        _uplo = builder.bitcast(uplo, u64)

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

        fnty = llvmir.FunctionType(
            utils.LLVMTypes.void_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                u64,
                u64,
                u64,
                cgutils.voidptr_t,
            ],
        )

        fn = cgutils.get_or_insert_function(mod, fnty, lapack_func_name)
        builder.call(fn, [_a_ptr, _v_ptr, _w_ptr, _lda, _n, _uplo, _queue_ref])

        if qref_payload.py_dpctl_sycl_queue_addr:
            qref_payload.pyapi.decref(qref_payload.py_dpctl_sycl_queue_addr)

        return None

    return signature, codegen


@overload(dpnp.linalg.eigh, prefer_literal=True)
def ol_dpnp_linalg_eigh(a, UPLO="L"):
    _a_dtype = a.dtype

    # if isinstance(_a_dtype, Complex):
    #     raise ValueError(
    #         "dpnp.linalg.eigh() overload doesn't "
    #         + "support complex values yet."
    #     )

    if isinstance(_a_dtype, Integer):
        raise ValueError(
            "dpnp.linalg.eigh() overload doesn't "
            + "support integer values yet."
        )

    if _a_dtype.bitwidth < 32:
        raise ValueError(
            "Less than 32 bit array type is "
            + "unsupported in dpnp.linalg.eigh()."
        )

    _upper_lower = {"U": 1, "L": -1}
    _usm_type = a.usm_type
    _sycl_queue = a.queue
    _order = "C" if a.is_c_contig else "F"
    _v_dtype, _w_dtype = _parse_dtypes(a)

    # We are flipping UPLO values since we can't use dpnp.asfarray()
    # or dpnp.copy() in impl(), i.e. those overloads aren't implemented yet.
    #
    # The side-effect being some of the eigenvectors might have flipped
    # signs, when 'a' is C-contiguous. However, mathematically signs of
    # eigenvectors don't matter.
    _uplo = _upper_lower[UPLO] if _order == "F" else (-1 * _upper_lower[UPLO])

    def impl(
        a,
        UPLO="L",
    ):
        if a.ndim > 2:
            raise NotImplementedError("a.ndim > 2 is not supported yet.")
        else:
            if a.ndim < 2:
                raise ValueError("Input array 'a' must have two dimensions.")

            lda, n = a.shape[-2], a.shape[-1]
            if lda != n:
                raise ValueError(
                    "Last 2 dimensions of the array must be square."
                )

            # We are doing transpose since we can't use dpnp.asfarray()
            # or dpnp.copy() here, i.e. those overloads aren't implemented yet.
            #
            # The side-effect being some of the eigenvectors might have flipped
            # signs, when 'a' is C-contiguous. However, mathematically signs of
            # eigenvectors don't matter.
            #
            # TODO: Implement overloads for dpnp.asfarray() and dpnp.copy()
            _a = a.T if _order != "F" else a

            v = dpnp.empty(
                (lda, n), dtype=_v_dtype, order="F", usm_type=_usm_type
            )
            w = dpnp.empty(
                (1, n), dtype=_w_dtype, order="F", usm_type=_usm_type
            )

            impl_dpnp_linalg_eigh(_a, v, w, lda, n, _uplo, _sycl_queue)

            return (w, v)

    return impl
