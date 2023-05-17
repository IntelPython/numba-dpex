# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from llvmlite.ir.types import DoubleType, FloatType
from numba import types
from numba.core.typing import signature
from numba.extending import intrinsic
from numba.np.arrayobj import (
    _empty_nd_impl,
    _parse_empty_args,
    _parse_empty_like_args,
    get_itemsize,
)

from numba_dpex.core.runtime import context as dpexrt


def alloc_empty_arrayobj(context, builder, sig, args, is_like=False):
    """Construct an empty numba.np.arrayobj.make_array.<locals>.ArrayStruct

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        sig (numba.core.typing.templates.Signature): A numba's function
            signature object.
        args (tuple): A tuple of args to be parsed as the arguments of
            an np.empty(), np.zeros() or np.ones() call.
        is_like (bool, optional): Decides on how to parse the args.
            Defaults to False.

    Returns:
        tuple(numba.np.arrayobj.make_array.<locals>.ArrayStruct,
            numba_dpex.core.types.dpnp_ndarray_type.DpnpNdArray):
                A tuple of allocated array and constructed array type info
                in DpnpNdArray.
    """

    arrtype = (
        _parse_empty_like_args(context, builder, sig, args)
        if is_like
        else _parse_empty_args(context, builder, sig, args)
    )
    ary, queue = _empty_nd_impl(context, builder, *arrtype)

    return ary, arrtype, queue


def fill_arrayobj(context, builder, ary, arrtype, queue_ref, fill_value):
    """Fill a numba.np.arrayobj.make_array.<locals>.ArrayStruct
        with a specified value.

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        ary (numba.np.arrayobj.make_array.<locals>.ArrayStruct): A numba
            arrystruct allocated by numba's `make_array()` function.
        arrtype (tuple): Parsed arguments by numba's `_parse_empty_args`
            like functions for different numpy/dpnp methods, e.g. `zeros()`,
            `ones()`, `empty()`, and their corresponding `_like()` methods.
        fill_value (llvmlite.ir.values.Argument): An LLVMLite IR `Argument`
            object that specifies the values to be filled in.

    Returns:
        tuple(numba.np.arrayobj.make_array.<locals>.ArrayStruct,
            numba_dpex.core.types.dpnp_ndarray_type.DpnpNdArray):
                A tuple of allocated array and constructed array type info
                in DpnpNdArray.
    """

    itemsize = context.get_constant(
        types.intp, get_itemsize(context, arrtype[0])
    )

    if isinstance(fill_value.type, DoubleType) or isinstance(
        fill_value.type, FloatType
    ):
        value_is_float = context.get_constant(types.boolean, 1)
    else:
        value_is_float = context.get_constant(types.boolean, 0)

    if isinstance(arrtype[0].dtype, types.scalars.Float):
        dest_is_float = context.get_constant(types.boolean, 1)
    else:
        dest_is_float = context.get_constant(types.boolean, 0)

    # Do a bitcast of the input to a 64-bit int.
    value = builder.bitcast(fill_value, llvmir.IntType(64))

    dpexrtCtx = dpexrt.DpexRTContext(context)
    dpexrtCtx.meminfo_fill(
        builder,
        ary.meminfo,
        itemsize,
        dest_is_float,
        value_is_float,
        value,
        queue_ref,
    )
    return ary, arrtype


@intrinsic
def intrin_usm_alloc(typingctx, allocsize, usm_type, queue):
    """Intrinsic to call into the allocator for Array"""

    def codegen(context, builder, signature, args):
        [allocsize, usm_type, queue] = args
        dpexrtCtx = dpexrt.DpexRTContext(context)
        meminfo = dpexrtCtx.meminfo_alloc(builder, allocsize, usm_type, queue)
        return meminfo

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, usm_type, queue)
    return sig, codegen


@intrinsic
def impl_dpnp_empty(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.empty().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, _, _ = alloc_empty_arrayobj(context, builder, sig, args)
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_zeros(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.zeros().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args
        )
        fill_value = context.get_constant(types.intp, 0)
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_ones(
    ty_context,
    ty_shape,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.ones().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_shape,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args
        )
        fill_value = context.get_constant(types.intp, 1)
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_full(
    ty_context,
    ty_shape,
    ty_fill_value,
    ty_dtype,
    ty_order,
    ty_like,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.full().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array.
        ty_fill_value (numba.core.types.scalars): One of the Numba scalar
            types.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_like (numba.core.types.npytypes.Array): Numba type for array.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    signature = ty_retty(
        ty_shape,
        ty_fill_value,
        ty_dtype,
        ty_order,
        ty_like,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args
        )
        fill_value = context.get_argument_value(builder, sig.args[1], args[1])
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return signature, codegen


@intrinsic
def impl_dpnp_empty_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.empty_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, _, _ = alloc_empty_arrayobj(
            context, builder, sig, args, is_like=True
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_zeros_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.zeros_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args, is_like=True
        )
        fill_value = context.get_constant(types.intp, 0)
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_ones_like(
    ty_context,
    ty_x1,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.ones_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    sig = ty_retty(
        ty_x1,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args, is_like=True
        )
        fill_value = context.get_constant(types.intp, 1)
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_full_like(
    ty_context,
    ty_x1,
    ty_fill_value,
    ty_dtype,
    ty_order,
    ty_subok,
    ty_shape,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.full_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x1 (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_fill_value (numba.core.types.scalars): One of the Numba scalar
            types.
        ty_dtype (numba.core.types.functions.NumberClass): Numba type for
            dtype.
        ty_order (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_subok (numba.core.types.scalars.Boolean): Numba type class for
            subok.
        ty_shape (numba.core.types.scalars.Integer or
            numba.core.types.containers.UniTuple): Numba type for the shape
            of the array. Not supported.
        ty_device (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_usm_type (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_sycl_queue (numba.core.types.misc.UnicodeType): UnicodeType
            from numba for strings.
        ty_retty_ref (numba.core.types.abstract.TypeRef): Reference to
            a type from numba, used when a type is passed as a value.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    ty_retty = ty_retty_ref.instance_type
    signature = ty_retty(
        ty_x1,
        ty_fill_value,
        ty_dtype,
        ty_order,
        ty_subok,
        ty_shape,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, args):
        ary, arrtype, queue_ref = alloc_empty_arrayobj(
            context, builder, sig, args, is_like=True
        )
        fill_value = context.get_argument_value(builder, sig.args[1], args[1])
        ary, _ = fill_arrayobj(
            context, builder, ary, arrtype, queue_ref, fill_value
        )
        return ary._getvalue()

    return signature, codegen
