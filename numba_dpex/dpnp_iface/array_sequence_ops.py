import math
from collections import namedtuple

import dpctl.tensor as dpt
import dpnp
import numba
import numpy as np
from dpctl.tensor._ctors import _coerce_and_infer_dt
from llvmlite import ir as llvmir
from numba import errors, types
from numba.core import cgutils
from numba.core.types.misc import NoneType, UnicodeType
from numba.core.types.scalars import (
    Boolean,
    Complex,
    Float,
    Integer,
    IntegerLiteral,
)
from numba.core.typing.templates import Signature
from numba.extending import intrinsic, overload

import numba_dpex.utils as utils
from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpnpNdArray
from numba_dpex.dpnp_iface._intrinsic import (
    _ArgTyAndValue,
    _empty_nd_impl,
    _get_queue_ref,
    alloc_empty_arrayobj,
)
from numba_dpex.dpnp_iface.arrayobj import (
    _parse_device_filter_string,
    _parse_dim,
    _parse_dtype,
    _parse_usm_type,
)

_QueueRefPayload = namedtuple(
    "QueueRefPayload", ["queue_ref", "py_dpctl_sycl_queue_addr", "pyapi"]
)


def _is_any_float_type(value):
    return (
        type(value) == float
        or isinstance(value, np.floating)
        or isinstance(value, Float)
    )


def _is_any_int_type(value):
    return (
        type(value) == int
        or isinstance(value, np.integer)
        or isinstance(value, Integer)
    )


def _is_any_complex_type(value):
    return np.iscomplex(value) or isinstance(value, Complex)


def _compute_bitwidth(value):
    print("_compute_bitwidth(): type(value) =", type(value))
    if (
        isinstance(value, Float)
        or isinstance(value, Integer)
        or isinstance(value, Complex)
    ):
        return value.bitwidth
    elif (
        isinstance(value, np.floating)
        or isinstance(value, np.integer)
        or np.iscomplex(value)
    ):
        return value.itemsize * 8
    elif type(value) == float or type(value) == int:
        return 64
    elif type(value) == complex:
        return 128
    else:
        msg = "dpnp_iface.array_sequence_ops._compute_bitwidth(): Unknwon type."
        raise errors.NumbaValueError(msg)


def _parse_dtype_from_range(start, stop, step):
    max_bw = max(
        _compute_bitwidth(start),
        _compute_bitwidth(stop),
        _compute_bitwidth(step),
    )
    if (
        _is_any_complex_type(start)
        or _is_any_complex_type(stop)
        or _is_any_complex_type(step)
    ):
        return (
            numba.from_dtype(dpnp.complex128)
            if max_bw == 128
            else numba.from_dtype(dpnp.complex64)
        )
    elif (
        _is_any_float_type(start)
        or _is_any_float_type(stop)
        or _is_any_float_type(step)
    ):
        if max_bw == 64:
            return numba.from_dtype(dpnp.float64)
        elif max_bw == 32:
            return numba.from_dtype(dpnp.float32)
        elif max_bw == 16:
            return numba.from_dtype(dpnp.float16)
        else:
            return numba.from_dtype(dpnp.float)
    elif (
        _is_any_int_type(start)
        or _is_any_int_type(stop)
        or _is_any_int_type(step)
    ):
        if max_bw == 64:
            return numba.from_dtype(dpnp.int64)
        elif max_bw == 32:
            return numba.from_dtype(dpnp.int32)
        else:
            return numba.from_dtype(dpnp.int)
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


def _get_constant(context, dtype, bitwidth, value):
    if isinstance(dtype, Integer):
        if bitwidth == 64:
            return context.get_constant(types.int64, value)
        elif bitwidth == 32:
            return context.get_constant(types.int32, value)
        elif bitwidth == 16:
            return context.get_constant(types.int16, value)
        elif bitwidth == 8:
            return context.get_constant(types.int8, value)
    elif isinstance(dtype, Float):
        if bitwidth == 64:
            return context.get_constant(types.float64, value)
        elif bitwidth == 32:
            return context.get_constant(types.float32, value)
        elif bitwidth == 16:
            return context.get_constant(types.float16, value)
    elif isinstance(dtype, Complex):
        if bitwidth == 128:
            return context.get_constant(types.complex128, value)
        elif bitwidth == 64:
            return context.get_constant(types.complex64, value)
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._get_constant():"
            + " Couldn't infer type for the requested constant."
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
        mod = builder.module

        start_ir, stop_ir, step_ir, dtype_ir, queue_ir = (
            args[0],
            args[1],
            args[2],
            args[3],
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

        # b = llvmir.IntType(1) # noqa: E800
        # u32 = llvmir.IntType(32)  # noqa: E800
        u64 = llvmir.IntType(64)
        # f32 = llvmir.FloatType()  # noqa: E800
        f64 = llvmir.DoubleType()  # noqa: E800
        # zero_u32 = context.get_constant(types.int32, 0)   # noqa: E800
        # zero_u64 = context.get_constant(types.int64, 0)   # noqa: E800
        # zero_f32 = context.get_constant(types.float32, 0) # noqa: E800
        zero_f64 = context.get_constant(types.float64, 0)
        # one_u32 = context.get_constant(types.int32, 1)    # noqa: E800
        # one_u64 = context.get_constant(types.int64, 1)    # noqa: E800
        # one_f32 = context.get_constant(types.float32, 1)  # noqa: E800
        one_f64 = context.get_constant(types.float64, 1)

        # ftype = _get_llvm_type(dtype_arg_type.dtype)  # noqa: E800
        # utype = _get_llvm_type(dtype_arg_type.dtype)  # noqa: E800
        # one = _get_constant(  # noqa: E800
        #     context, dtype_arg_type.dtype, dtype_arg_type.dtype.bitwidth, 1   # noqa: E800
        # ) # noqa: E800
        # zero = _get_constant( # noqa: E800
        #     context, dtype_arg_type.dtype, dtype_arg_type.dtype.bitwidth, 0   # noqa: E800
        # ) # noqa: E800

        print(
            f"start_ir = {start_ir}, "
            + f"start_ir.type = {start_ir.type}, "
            + f"type(start_ir.type) = {type(start_ir.type)}"
        )
        print(
            f"step_ir = {step_ir}, "
            + f"step_ir.type = {step_ir.type}, "
            + f"type(step_ir.type) = {type(step_ir.type)}"
        )
        print(
            f"stop_ir = {stop_ir}, "
            + f"stop_ir.type = {stop_ir.type}, "
            + f"type(stop_ir.type) = {type(stop_ir.type)}"
        )

        # Sanity check:
        # if stop is pointing to a null
        #    start <- 0
        #    stop <- 1
        # if step is pointing to a null
        #    step <- 1
        # TODO: do this either in LLVMIR or outside of intrinsic
        print("type(stop_arg_type) =", type(stop_arg_type))
        print("type(step_arg_type) =", type(step_arg_type))
        if isinstance(stop_arg_type, NoneType):
            start_ir = zero_f64
            stop_ir = one_f64
        if isinstance(step_arg_type, NoneType):
            step_ir = one_f64

        if isinstance(start_arg_type, Integer) and isinstance(
            dtype_arg_type.dtype, Float
        ):
            if start_arg_type.signed:
                start_ir = builder.sitofp(start_ir, f64)
                step_ir = builder.sitofp(step_ir, f64)
            else:
                start_ir = builder.uitofp(start_ir, f64)
                step_ir = builder.uitofp(step_ir, f64)

        print(
            f"start_ir = {start_ir}, "
            + f"start_ir.type = {start_ir.type}, "
            + f"type(start_ir.type) = {type(start_ir.type)}"
        )
        print(
            f"step_ir = {step_ir}, "
            + f"step_ir.type = {step_ir.type}, "
            + f"type(step_ir.type) = {type(step_ir.type)}"
        )
        print(
            f"stop_ir = {stop_ir}, "
            + f"stop_ir.type = {stop_ir.type}, "
            + f"type(stop_ir.type) = {type(stop_ir.type)}"
        )
        print(
            f"dtype_ir = {dtype_ir}, "
            + f"dtype_ir.type = {dtype_ir.type}, "
            + f"dtype_arg_type = {dtype_arg_type}, "
            + f"dtype_arg_type.dtype = {dtype_arg_type.dtype}, "
            + f"dtype_arg_type.dtype.bitwidth = {dtype_arg_type.dtype.bitwidth}"
        )

        # Get SYCL Queue ref
        sycl_queue_arg = _ArgTyAndValue(queue_arg_type, queue_ir)
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )

        with builder.goto_entry_block():
            start_ptr = cgutils.alloca_once(builder, start_ir.type)
            step_ptr = cgutils.alloca_once(builder, step_ir.type)

        builder.store(start_ir, start_ptr)
        builder.store(step_ir, step_ptr)

        start_vptr = builder.bitcast(start_ptr, cgutils.voidptr_t)
        step_vptr = builder.bitcast(step_ptr, cgutils.voidptr_t)

        ll = builder.sitofp(start_ir, f64)
        ul = builder.sitofp(stop_ir, f64)
        d = builder.sitofp(step_ir, f64)

        # Doing ceil(a,b) = (a-1)/b + 1 to avoid overflow
        t = builder.fptosi(
            builder.fadd(
                builder.fdiv(builder.fsub(builder.fsub(ul, ll), one_f64), d),
                one_f64,
            ),
            u64,
        )

        # Allocate an empty array
        ary = _empty_nd_impl(
            context, builder, sig.return_type, [t], qref_payload.queue_ref
        )

        # Convert into void*
        arrystruct_vptr = builder.bitcast(ary._getpointer(), cgutils.voidptr_t)

        # Function parameters
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
            mod, fnty, "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence"
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
    print("start =", start, ", type(start) =", type(start))
    print("stop =", stop, ", type(stop) =", type(stop))
    print("step =", step, ", type(step) =", type(step))
    print("dtype =", dtype, ", type(dtype) =", type(dtype))
    print("---")

    if stop is None:
        start = 0
        stop = 1
    if step is None:
        step = 1

    print("start =", start, ", type(start) =", type(start))
    print("stop =", stop, ", type(stop) =", type(stop))
    print("step =", step, ", type(step) =", type(step))
    print("***")

    _dtype = (
        _parse_dtype(dtype)
        if dtype is not None
        else _parse_dtype_from_range(start, stop, step)
    )
    print("_dtype =", _dtype, ", type(_dtype) =", type(_dtype))

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
