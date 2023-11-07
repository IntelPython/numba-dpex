# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import dpnp
import numba
from llvmlite import ir as llvmir
from llvmlite.ir import types as llvmirtypes
from numba import errors, types
from numba.core import cgutils
from numba.core.types.misc import NoneType
from numba.core.types.scalars import Boolean, Complex, Float, Integer
from numba.extending import intrinsic, overload
from numba.np.numpy_support import is_nonelike

import numba_dpex.utils as utils
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.dpnp_iface._intrinsic import (
    _ArgTyAndValue,
    _empty_nd_impl,
    _get_queue_ref,
)
from numba_dpex.dpnp_iface.arrayobj import (
    _parse_device_filter_string,
    _parse_dtype,
    _parse_usm_type,
)

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)


def _is_float_type(start, stop, step):
    return (
        type(start) == float
        or type(stop) == float
        or type(step) == float
        or isinstance(start, Float)
        or isinstance(stop, Float)
        or isinstance(step, Float)
    )


def _is_int_type(start, stop, step):
    return (
        type(start) == int
        or type(stop) == int
        or type(step) == int
        or isinstance(start, Integer)
        or isinstance(stop, Integer)
        or isinstance(step, Integer)
    )


def _is_complex_type(start, stop, step):
    return (
        isinstance(start, Complex)
        or isinstance(stop, Complex)
        or isinstance(step, Complex)
    )


def _parse_dtype_from_range(start, stop, step):
    if _is_complex_type(start, stop, step):
        return numba.from_dtype(dpnp.complex128)
    elif _is_float_type(start, stop, step):
        return numba.from_dtype(dpnp.float64)
    elif _is_int_type(start, stop, step):
        return numba.from_dtype(dpnp.int64)
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._parse_dtype_from_range(): "
            + "Types couldn't be inferred for (start, stop, step)."
        )
        raise errors.NumbaValueError(msg)


def _get_llvm_type(numba_type):
    if isinstance(numba_type, Integer):
        return llvmir.IntType(numba_type.bitwidth)
    elif isinstance(numba_type, Float):
        if numba_type.bitwidth == 64:
            return llvmir.DoubleType()
        elif numba_type.bitwidth == 32:
            return llvmir.FloatType()
        elif numba_type.bitwidth == 16:
            return llvmir.HalfType()
        else:
            msg = (
                "dpnp_iface.array_sequence_ops._get_llvm_type(): "
                + f"Incompatible bitwidth in {numba_type}."
            )
            raise errors.NumbaTypeError(msg)
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._get_llvm_type(): "
            + "Incompatible numba type."
        )
        raise errors.NumbaTypeError(msg)


def _get_dst_typeid(dtype):
    if isinstance(dtype, Boolean):
        return 0
    elif isinstance(dtype, Integer):
        if dtype.bitwidth == 8:
            return 1 if dtype.signed else 2
        elif dtype.bitwidth == 16:
            return 3 if dtype.signed else 4
        elif dtype.bitwidth == 32:
            return 5 if dtype.signed else 6
        elif dtype.bitwidth == 64:
            return 7 if dtype.signed else 8
        else:
            msg = (
                "dpnp_iface.array_sequence_ops._get_dst_typeid(): "
                + f"Couldn't map {dtype} to dst_index."
            )
            raise errors.NumbaValueError(msg)
    elif isinstance(dtype, Float):
        if dtype.bitwidth == 16:
            return 9
        elif dtype.bitwidth == 32:
            return 10
        elif dtype.bitwidth == 64:
            return 11
        else:
            msg = (
                "dpnp_iface.array_sequence_ops._get_dst_typeid(): "
                + f"Couldn't map {dtype} to dst_index."
            )
            raise errors.NumbaValueError(msg)
    elif isinstance(dtype, Complex):
        if dtype.bitwidth == 64:
            return 12
        elif dtype.bitwidth == 128:
            return 13
        else:
            msg = (
                "dpnp_iface.array_sequence_ops._get_dst_typeid(): "
                + f"Couldn't map {dtype} to dst_index."
            )
            raise errors.NumbaValueError(msg)
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._get_dst_typeid(): "
            + f"Unknown numba type {dtype}"
        )
        raise errors.NumbaTypeError(msg)


def _round(builder, src, src_type):
    return_type = (
        llvmirtypes.DoubleType()
        if src_type.bitwidth == 64
        else (
            llvmirtypes.FloatType()
            if src_type.bitwidth == 32
            else llvmirtypes.HalfType()
        )
    )
    round = builder.module.declare_intrinsic("llvm.round", [return_type])
    src = builder.call(round, [src])
    return src


def _is_fraction(builder, src, src_type):
    if isinstance(src_type, Float):
        return_type = (
            llvmirtypes.DoubleType()
            if src_type.bitwidth == 64
            else (
                llvmirtypes.FloatType()
                if src_type.bitwidth == 32
                else llvmirtypes.HalfType()
            )
        )
        llvm_fabs = builder.module.declare_intrinsic("llvm.fabs", [return_type])
        src_abs = builder.call(llvm_fabs, [src])
        ret = True
        is_lto = builder.fcmp_ordered(">=", src_abs, src.type(1.0))
        with builder.if_then(is_lto):
            ret = False
        return ret


def _normalize(builder, src, src_type, dest_type, rounding=False):
    dest_llvm_type = _get_llvm_type(dest_type)
    if isinstance(src_type, Integer) and isinstance(dest_type, Integer):
        if src_type.bitwidth < dest_type.bitwidth:
            return builder.zext(src, dest_llvm_type)
        elif src_type.bitwidth > dest_type.bitwidth:
            return builder.trunc(src, dest_llvm_type)
        else:
            return src
    elif isinstance(src_type, Integer) and isinstance(dest_type, Float):
        if src_type.signed:
            return builder.sitofp(src, dest_llvm_type)
        else:
            return builder.uitofp(src, dest_llvm_type)
    elif isinstance(src_type, Float) and isinstance(dest_type, Integer):
        if rounding:
            src = _round(builder, src, src_type)
        if dest_type.signed:
            return builder.fptosi(src, dest_llvm_type)
        else:
            return builder.fptoui(src, dest_llvm_type)
    elif isinstance(src_type, Float) and isinstance(dest_type, Float):
        if src_type.bitwidth < dest_type.bitwidth:
            return builder.fpext(src, dest_llvm_type)
        elif src_type.bitwidth > dest_type.bitwidth:
            return builder.fptrunc(src, dest_llvm_type)
        else:
            return src
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._normalize(): "
            + f"{src}[{src_type}] is neither a "
            + "'numba.core.types.scalars.Float' "
            + "nor an 'numba.core.types.scalars.Integer'."
        )
        raise errors.NumbaTypeError(msg)


def _compute_array_length_ir(
    builder,
    start_ir,
    stop_ir,
    step_ir,
    start_arg_type,
    stop_arg_type,
    step_arg_type,
):
    lb = _normalize(builder, start_ir, start_arg_type, types.float64)
    ub = _normalize(builder, stop_ir, stop_arg_type, types.float64)
    n = _normalize(builder, step_ir, step_arg_type, types.float64)

    llvm_ceil = builder.module.declare_intrinsic(
        "llvm.ceil", [llvmirtypes.DoubleType()]
    )

    array_length_ir = builder.fptosi(
        builder.call(
            llvm_ceil,
            [builder.fdiv(builder.fsub(ub, lb), n)],
        ),
        llvmir.IntType(64),
    )

    return array_length_ir


@intrinsic
def impl_dpnp_arange(
    ty_context,
    ty_start,
    ty_stop,
    ty_step,
    ty_dtype,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_ret_ty,
):
    ty_retty_ = ty_ret_ty.instance_type
    signature = ty_retty_(
        ty_start,
        ty_stop,
        ty_step,
        ty_dtype,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_ret_ty,
    )

    sycl_queue_arg_pos = -2

    def codegen(context, builder, sig, args):
        # Rename variables for easy coding
        start_ir, stop_ir, step_ir, queue_ir = (
            args[0],
            args[1],
            args[2],
            args[sycl_queue_arg_pos],
        )
        (
            start_arg_type,
            stop_arg_type,
            step_arg_type,
            dtype_arg_type,
            queue_arg_type,
        ) = (
            sig.args[0],
            sig.args[1],
            sig.args[2],
            sig.args[3],
            sig.args[sycl_queue_arg_pos],
        )

        # Get SYCL Queue ref
        sycl_queue_arg = _ArgTyAndValue(queue_arg_type, queue_ir)
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )

        # Sanity check:
        # if stop is pointing to a null
        #    start <- 0
        #    stop <- 1
        # if step is pointing to a null
        #    step <- 1
        if isinstance(stop_arg_type, NoneType):
            start_ir = context.get_constant(start_arg_type, 0)
            stop_ir = context.get_constant(start_arg_type, 1)
            stop_arg_type = start_arg_type
        if isinstance(step_arg_type, NoneType):
            step_ir = context.get_constant(start_arg_type, 1)
            step_arg_type = start_arg_type

        # Keep note if either start or stop is in (-1.0, 0.0] or [0.0, 1.0)
        round_step = not (
            _is_fraction(builder, start_ir, start_arg_type)
            and _is_fraction(builder, stop_ir, stop_arg_type)
        )

        # Allocate an empty array
        len = _compute_array_length_ir(
            builder,
            start_ir,
            stop_ir,
            step_ir,
            start_arg_type,
            stop_arg_type,
            step_arg_type,
        )
        ary = _empty_nd_impl(
            context, builder, sig.return_type, [len], qref_payload.queue_ref
        )
        # Convert into void*
        arrystruct_vptr = builder.bitcast(ary._getpointer(), cgutils.voidptr_t)

        # Extend or truncate input values w.r.t. destination array type
        start_ir = _normalize(
            builder, start_ir, start_arg_type, dtype_arg_type.dtype
        )
        stop_ir = _normalize(
            builder, stop_ir, stop_arg_type, dtype_arg_type.dtype
        )
        step_ir = _normalize(
            builder,
            step_ir,
            step_arg_type,
            dtype_arg_type.dtype,
            rounding=round_step,
        )

        # After normalization, their arg_types will change
        start_arg_type = dtype_arg_type.dtype
        stop_arg_type = dtype_arg_type.dtype
        step_arg_type = dtype_arg_type.dtype

        # Construct function parameters
        with builder.goto_entry_block():
            start_ptr = cgutils.alloca_once(builder, start_ir.type)
            step_ptr = cgutils.alloca_once(builder, step_ir.type)
        builder.store(start_ir, start_ptr)
        builder.store(step_ir, step_ptr)
        start_vptr = builder.bitcast(start_ptr, cgutils.voidptr_t)
        step_vptr = builder.bitcast(step_ptr, cgutils.voidptr_t)
        ndim = context.get_constant(types.intp, 1)
        is_c_contguous = context.get_constant(types.int8, 1)
        typeid_index = _get_dst_typeid(dtype_arg_type.dtype)
        dst_typeid = context.get_constant(types.intp, typeid_index)

        # Function signature
        fnty = llvmir.FunctionType(
            utils.LLVMTypes.int64_ptr_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                _get_llvm_type(types.intp),
                _get_llvm_type(types.int8),
                _get_llvm_type(types.intp),
                cgutils.voidptr_t,
            ],
        )

        # Kernel call
        fn = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence",
        )
        builder.call(
            fn,
            [
                start_vptr,
                step_vptr,
                arrystruct_vptr,
                ndim,
                is_c_contguous,
                dst_typeid,
                qref_payload.queue_ref,
            ],
        )

        return ary._getvalue()

    return signature, codegen


@overload(dpnp.arange, prefer_literal=True)
def ol_dpnp_arange(
    start,
    stop=None,
    step=1,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    if isinstance(start, Complex) or (
        not is_nonelike(dtype) and isinstance(dtype.dtype, Complex)
    ):
        raise errors.NumbaNotImplementedError(
            "Complex type is not supported yet."
        )
    if isinstance(start, Boolean) or (
        not is_nonelike(dtype) and isinstance(dtype.dtype, Boolean)
    ):
        raise errors.NumbaTypeError(
            "Boolean is not supported by dpnp.arange()."
        )

    if is_nonelike(stop):
        start = 0 if type(start) == int or isinstance(start, Integer) else 0.0
        stop = 1 if type(start) == int or isinstance(start, Integer) else 1.0
    if is_nonelike(step):
        step = 1 if type(start) == int or isinstance(start, Integer) else 1.0

    _dtype = (
        _parse_dtype(dtype)
        if not is_nonelike(dtype)
        else _parse_dtype_from_range(start, stop, step)
    )
    _device = _parse_device_filter_string(device) if device else None
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"

    ret_ty = DpnpNdArray(
        ndim=1,
        layout="C",
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=sycl_queue,
    )

    if ret_ty:

        def impl(
            start,
            stop=None,
            step=1,
            dtype=None,
            device=None,
            usm_type="device",
            sycl_queue=None,
        ):
            return impl_dpnp_arange(
                start,
                stop,
                step,
                _dtype,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.arange({start}, {stop}, {step}, ...)."
        )
