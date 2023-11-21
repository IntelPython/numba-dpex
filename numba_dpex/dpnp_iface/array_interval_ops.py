# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import dpnp
import numba
from llvmlite import ir as llvmir
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


def _match_type(args, numba_type, basic_type=None):
    """Checks if one of the paramteres in `args` is any of the
    `numba.core.types.scalars.*` type.

    Args:
        args (list): A list of python basic type or `numba.core.types.scalars.*`
            type.
        numba_type ('numba.core.types.scalars.*'): The `numba` type to which
            `args[i]` would be compared with.
        basic_type (int, float): The basic type to which `args[i]` would be
            compared with.

    Returns:
        bool: True if one of `start`, `stop` or `step` is
            `numba.core.types.scalars.Float` type.
    """
    for arg in args:
        if (basic_type and type(arg) == basic_type) or isinstance(
            arg, numba_type
        ):
            return True
    return False


def _parse_dtype_from_range(range):
    """Infer `dtype` of the output tensor from input `numba` types in `range`.

    The `range` is a 3-tuple of `start`, `stop` and `step`(or `num`) values to
    denote the range of the interval. Each value is of `numba.core.types.scalars.*`
    type.

    Args:
        range (tuple): A 3-tuple of `start`, `step` and `step`/`num` value to
            denote the range of the interval. Each value is of
            `numba.core.types.scalars.*` type.

    Raises:
        errors.NumbaTypeError: If types couldn't be inferred for `start`,
            `stop`, and/or `step`(or `num`).

    Returns:
        numba.core.types.scalars.*: Infered `dtype` for the output tensor.
    """
    if _match_type(range, Complex):
        return numba.from_dtype(dpnp.complex_)
    elif _match_type(range, Float, float):
        return numba.from_dtype(dpnp.float)
    elif _match_type(range, Integer, int):
        return numba.from_dtype(dpnp.int)
    else:
        msg = (
            "dpnp_iface.array_sequence_ops._parse_dtype_from_range(): "
            + "Types couldn't be inferred for (start, stop, step or num)."
        )
        raise errors.NumbaTypeError(msg)


def _get_llvm_type(numba_type):
    """Returns `llvmlite.ir.types` from a corresponding
        `numba.core.types.scalars.*` `numba` type

    Args:
        numba_type (numba.core.types.scalars.*): The input `numba` type.

    Raises:
        errors.NumbaTypeError: If `numba_type` is
            `numba.core.types.scalars.Integer` and incompatible bitwidth.
        errors.NumbaTypeError: If `numba_type` is neither
            `numba.core.types.scalars.Integer` nor
            `numba.core.types.scalars.Float`

    Returns:
        llvmlite.ir.types: The LLVM IR type of the corresponding `numba` type.
    """
    if isinstance(numba_type, Integer):
        return llvmir.IntType(numba_type.bitwidth)
    elif isinstance(numba_type, Boolean):
        return llvmir.IntType(1)
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
    """Get the corresponding index for `interval_step_dispatch_vector` from
        `numba` type

    Args:
        dtype (numba.core.types.scalars.*): The `numba` type to index the
            function in `interval_step_dispatch_vector` in `intervals.cpp`.

    Raises:
        errors.NumbaValueError: If `dtype` is `numba.core.types.scalars.Integer`
            with an incompatible bitwidth.
        errors.NumbaValueError: If `dtype` is `numba.core.types.scalars.Float`
            with an incompatible bitwidth.
        errors.NumbaValueError: If `dtype` is `numba.core.types.scalars.Complex`
            with an incompatible bitwidth.
        errors.NumbaTypeError: If `dtype` is an unknown `numba.core.types`.

    Returns:
        int: Returns the type-id specified in the
            `dpx::rt::kernel::tensor::typenum_t`
    """
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
    """Round a value held by an LLVM IR instruction.

    Args:
        builder (llvmlite.ir.builder.IRBuilder): The LLVM IR Builder object.
        src (llvmlite.ir.*): LLVM IR instruction to hold input (source) type.
        src_type (numba.core.types.scalars.*): The `numba` type of the source
        type.

    Returns:
        llvmlite.ir.*: The LLVM IR instruction to get the rounded value.
    """
    return_type = (
        llvmir.DoubleType()
        if src_type.bitwidth == 64
        else (
            llvmir.FloatType() if src_type.bitwidth == 32 else llvmir.HalfType()
        )
    )
    round = builder.module.declare_intrinsic("llvm.round", [return_type])
    src = builder.call(round, [src])
    return src


def _normalize(builder, src, src_type, dest_type, float_to_int_rounding=False):
    """Converts/Normalizes two dissimilar types.

    Similar to type casting but it also handles bitwidth and rounding

    Args:
        builder (llvmlite.ir.builder.IRBuilder): The LLVM IR Builder object.
            src (llvmlite.ir.*): LLVM IR instruction to hold input (source)
            type.
        src_type (numba.core.types.scalars.*): The `numba` type of the source
            type.
        dest_type (numba.core.types.scalars.*): The `numba` type of the
            destination type.
        rounding (bool, optional): `True` if rounding needs to be done.
            Defaults to False.

    Raises:
        errors.NumbaTypeError: If `src_type` is neither a
            'numba.core.types.scalars.Float' nor an
            'numba.core.types.scalars.Integer'."

    Returns:
        llvmlite.ir.*: The LLVM IR instruction to get the casted value.
    """
    dest_llvm_type = _get_llvm_type(dest_type)
    if isinstance(src_type, Integer) and isinstance(dest_type, Integer):
        if src_type.bitwidth < dest_type.bitwidth:
            if src_type.signed:
                return builder.sext(src, dest_llvm_type)
            else:
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
        if float_to_int_rounding:
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
    context,
    builder,
    start_ir,
    stop_ir,
    step_ir,
    start_arg_type,
    stop_arg_type,
    step_arg_type,
):
    """LLVM IR generator to compute the length of the array.

    Args:
        builder (llvmlite.ir.builder.IRBuilder): The LLVM IR Builder object.
        start_ir (llvmlite.ir.*): LLVM IR to capture the `start` of the
            interval.
        stop_ir (llvmlite.ir.*): LLVM IR to capture the end of the interval.
        step_ir (llvmlite.ir.*): LLVM IR to capture the `step` of the interval.
        start_arg_type (numba.core.types.scalars.*): `numba` type for the
            `start`
        stop_arg_type (numba.core.types.scalars.*): `numba` type for the `stop`
        step_arg_type (numba.core.types.scalars.*): `numba` type for the `step`

    Returns:
        llvmlite.ir.instructions.*: The LLVM IR to contain the length of the
            array.
    """
    lb = _normalize(builder, start_ir, start_arg_type, types.float64)
    ub = _normalize(builder, stop_ir, stop_arg_type, types.float64)
    n = _normalize(builder, step_ir, step_arg_type, types.float64)

    llvm_ceil = builder.module.declare_intrinsic(
        "llvm.ceil", [llvmir.DoubleType()]
    )

    array_length_ir = builder.fptosi(
        builder.call(
            llvm_ceil,
            [builder.fdiv(builder.fsub(ub, lb), n)],
        ),
        llvmir.IntType(64),
    )

    is_ltez = builder.icmp_signed(
        "<=", array_length_ir, context.get_constant(types.int64, 0)
    )
    with builder.if_then(is_ltez, likely=False):
        # TODO: In these cases, numpy and dpnp return an empty array '[]'.
        #       We need to make changes to _empty_nd_impl() so that it can
        #       return an empty array like numpy and dpnp do.
        context.call_conv.return_user_exc(
            builder,
            ValueError,
            ("Impossible range, check your interval inputs.",),
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
    """A numba "intrinsic" function to inject code for dpnp.arange().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_start (numba.core.types.scalars.*): Numba type for the start of the
            interval.
        ty_stop (numba.core.types.scalars.*): Numba type for the end of the
            interval.
        ty_step (numba.core.types.scalars.Integer): Numba type for the step
            of the interval.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_ret_ty (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """
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

    def codegen(context, builder, sig, args):
        # Rename variables for easy coding
        start_ir, stop_ir, step_ir, queue_ir = (
            args[0],
            args[1],
            args[2],
            args[-2],
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
            sig.args[-2],
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

        # Allocate an empty array
        len = _compute_array_length_ir(
            context,
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
        # Round the step if int <-- float
        step_ir = _normalize(
            builder,
            step_ir,
            step_arg_type,
            dtype_arg_type.dtype,
            float_to_int_rounding=True,
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
        is_c_contguous = context.get_constant(types.boolean, True)
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
                _get_llvm_type(types.boolean),
                _get_llvm_type(types.intp),
                cgutils.voidptr_t,
            ],
        )

        # Kernel call
        fn = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_interval",
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
    """Implementation of an overload to support dpnp.arange() inside
    a dpjit function. Returns evenly spaced values within the half-open interval
    [start, stop) as a one-dimensional array.

    Args:
        start (numba.core.types.scalars.*): The start of the interval. If `stop`
            is specified, the start of interval (inclusive); otherwise, the end
            of the interval (exclusive). If `stop` is not specified, the default
            starting value is 0.
        stop (numba.core.types.scalars.*, optional): The end of the interval.
            Default: `None`.
        step (numba.core.types.scalars.*, optional): The distance between two
            adjacent elements (`out[i+1] - out[i]`). Must not be 0; may be
            negative, this results in an empty array if `stop >= start`.
            Default: 1.
        dtype (numba.core.types.scalars.*, optional): The output array data
            type. If `dtype` is `None`, the output array data type must be
            inferred from `start`, `stop` and `step`. If those are all integers,
            the output array `dtype` must be the default integer `dtype`; if
            one or more have type `float`, then the output array dtype must be
            the default real-valued floating-point data type. Default: `None`.
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.NumbaNotImplementedError: If `start` is
            `numba.core.types.scalars.Complex` type
        errors.NumbaTypeError: If `start` is `numba.core.types.scalars.Boolean`
            type
        errors.TypingError: If couldn't parse input types to dpnp.arange().

    Returns:
        function: Local function `impl_dpnp_arange()`.
    """

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
        start = 0 if _match_type([start], Integer, int) else 0.0
        stop = 1 if _match_type([start], Integer, int) else 1.0
    if is_nonelike(step):
        step = 1 if _match_type([start], Integer, int) else 1.0

    _dtype = (
        _parse_dtype(dtype)
        if not is_nonelike(dtype)
        else _parse_dtype_from_range([start, stop, step])
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


@intrinsic
def impl_dpnp_linspace(
    ty_context,
    ty_start,
    ty_stop,
    ty_num,
    ty_dtype,
    ty_endpoint,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_ret_ty,
):
    """A numba "intrinsic" function to inject code for dpnp.linspace().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_start (numba.core.types.scalars.Integer): Numba type for the start
            of the interval.
        ty_stop (numba.core.types.scalars.Integer): Numba type for the end
            of the interval.
        ty_step (numba.core.types.scalars.Integer): Numba type for the step
            of the interval.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_ret_ty (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """
    ty_retty_ = ty_ret_ty.instance_type
    signature = ty_retty_(
        ty_start,
        ty_stop,
        ty_num,
        ty_dtype,
        ty_endpoint,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_ret_ty,
    )

    def codegen(context, builder, sig, args):
        # Rename variables for easy coding
        start_ir, stop_ir, num_ir, endpoint_ir, queue_ir = (
            args[0],
            args[1],
            args[2],
            args[4],
            args[-2],
        )
        (
            start_arg_type,
            stop_arg_type,
            dtype_arg_type,
            queue_arg_type,
        ) = (
            sig.args[0],
            sig.args[1],
            sig.args[3],
            sig.args[-2],
        )

        # Get SYCL Queue ref
        sycl_queue_arg = _ArgTyAndValue(queue_arg_type, queue_ir)
        qref_payload: _QueueRefPayload = _get_queue_ref(
            context=context,
            builder=builder,
            returned_sycl_queue_ty=sig.return_type.queue,
            sycl_queue_arg=sycl_queue_arg,
        )

        # Allocate an empty array
        ary = _empty_nd_impl(
            context, builder, sig.return_type, [num_ir], qref_payload.queue_ref
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

        # After normalization, their arg_types will change
        start_arg_type = dtype_arg_type.dtype
        stop_arg_type = dtype_arg_type.dtype

        # Construct function parameters
        with builder.goto_entry_block():
            start_ptr = cgutils.alloca_once(builder, start_ir.type)
            stop_ptr = cgutils.alloca_once(builder, stop_ir.type)
        builder.store(start_ir, start_ptr)
        builder.store(stop_ir, stop_ptr)
        start_vptr = builder.bitcast(start_ptr, cgutils.voidptr_t)
        stop_vptr = builder.bitcast(stop_ptr, cgutils.voidptr_t)
        ndim = context.get_constant(types.intp, 1)
        is_c_contguous = context.get_constant(types.boolean, True)
        typeid_index = _get_dst_typeid(dtype_arg_type.dtype)
        dst_typeid = context.get_constant(types.intp, typeid_index)

        # Function signature
        fnty = llvmir.FunctionType(
            utils.LLVMTypes.int64_ptr_t,
            [
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                cgutils.voidptr_t,
                _get_llvm_type(types.boolean),
                _get_llvm_type(types.intp),
                _get_llvm_type(types.boolean),
                _get_llvm_type(types.intp),
                cgutils.voidptr_t,
            ],
        )

        # Kernel call
        fn = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_affine_interval",
        )
        builder.call(
            fn,
            [
                start_vptr,
                stop_vptr,
                arrystruct_vptr,
                endpoint_ir,
                ndim,
                is_c_contguous,
                dst_typeid,
                qref_payload.queue_ref,
            ],
        )

        return ary._getvalue()

    return signature, codegen


@overload(dpnp.linspace, prefer_literal=True)
def ol_dpnp_linspace(
    start,
    stop,
    num,
    dtype=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    retstep=False,
    axis=0,
):
    """Implementation of an overload to support dpnp.linspace() inside
    a dpjit function. Returns evenly spaced values within the half-open interval
    [start, stop) as a one-dimensional array.

    Args:
        start (numba.core.types.scalars.*): The start of the interval. If `stop`
            is specified, the start of interval (inclusive); otherwise, the end
            of the interval (exclusive). If `stop` is not specified, the default
            starting value is 0.
        stop (numba.core.types.scalars.*, optional): The end of the interval.
            Default: `None`.
        step (numba.core.types.scalars.*, optional): The distance between two
            adjacent elements (`out[i+1] - out[i]`). Must not be 0; may be
            negative, this results in an empty array if `stop >= start`.
            Default: 1.
        dtype (numba.core.types.scalars.*, optional): The output array data
            type. If `dtype` is `None`, the output array data type must be
            inferred from `start`, `stop` and `step`. If those are all integers,
            the output array `dtype` must be the default integer `dtype`; if
            one or more have type `float`, then the output array dtype must be
            the default real-valued floating-point data type. Default: `None`.
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.NumbaNotImplementedError: If `start` is
            `numba.core.types.scalars.Complex` type
        errors.NumbaTypeError: If `start` is `numba.core.types.scalars.Boolean`
            type
        errors.TypingError: If couldn't parse input types to dpnp.arange().

    Returns:
        function: Local function `impl_dpnp_linspace()`.
    """
    if not (
        (
            _match_type([start], Integer, int)
            or _match_type([start], Float, float)
        )
        and (
            _match_type([stop], Integer, int)
            or _match_type([stop], Float, float)
        )
    ):
        msg = (
            "Input data type is not supported."
            + " Please convert the input to"
            + " a scalar data type (int or float)."
        )
        raise errors.NumbaTypeError(msg)

    if not _match_type([num], Integer, int):
        msg = "'num' must be an int."
        raise errors.NumbaTypeError(msg)

    if retstep:
        msg = "'retstep=True' is not supported yet."
        raise errors.NumbaNotImplementedError(msg)
    if axis != 0:
        msg = "Nonzero 'axis' is not supported yet."
        raise errors.NumbaNotImplementedError(msg)

    _dtype = (
        _parse_dtype(dtype)
        if not is_nonelike(dtype)
        else numba.from_dtype(dpnp.float)
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
            stop,
            num,
            dtype=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
            endpoint=True,
            retstep=False,
            axis=0,
        ):
            return impl_dpnp_linspace(
                start,
                stop,
                num,
                _dtype,
                endpoint,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.linspace({start}, {stop}, {num}, ...)."
        )
