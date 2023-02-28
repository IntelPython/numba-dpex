# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir
from llvmlite.ir import Constant
from numba import types
from numba.core import cgutils
from numba.core.typing import signature
from numba.extending import intrinsic
from numba.np.arrayobj import (
    _empty_nd_impl,
    _parse_empty_args,
    _parse_empty_like_args,
    get_itemsize,
    make_array,
    populate_array,
)

from numba_dpex.core.runtime import context as dpexrt

from ..decorators import dpjit


@dpjit
# TODO: rename this to _call_allocator and see below
def _call_usm_allocator(arrtype, size, usm_type, device):
    """Trampoline to call the intrinsic used for allocation"""

    return arrtype._usm_allocate(size, usm_type, device)


# def _empty_nd_impl(context, builder, arrtype, shapes):
#     """Utility function used for allocating a new array during LLVM
#     code generation (lowering).

#     Given a target context, builder, array type, and a tuple or list
#     of lowered dimension sizes, returns a LLVM value pointing at a
#     Numba runtime allocated array.

#     Args:
#         context (numba.core.base.BaseContext): One of the class derived
#             from numba's BaseContext, e.g. CPUContext
#         builder (llvmlite.ir.builder.IRBuilder): IR builder object from
#             llvmlite.
#         arrtype (numba_dpex.core.types.dpnp_ndarray_type.DpnpNdArray):
#             An array type info to construct the actual array.
#         shapes (list): The dimension of the array.

#     Raises:
#         NotImplementedError: If the layout of the array is not known.

#     Returns:
#         numba.np.arrayobj.make_array.<locals>.ArrayStruct: The constructed
#             array.
#     """

#     arycls = make_array(arrtype)
#     ary = arycls(context, builder)

#     datatype = context.get_data_type(arrtype.dtype)
#     itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

#     # compute array length
#     arrlen = context.get_constant(types.intp, 1)
#     overflow = Constant(ir.IntType(1), 0)
#     for s in shapes:
#         arrlen_mult = builder.smul_with_overflow(arrlen, s)
#         arrlen = builder.extract_value(arrlen_mult, 0)
#         overflow = builder.or_(overflow, builder.extract_value(arrlen_mult, 1))

#     if arrtype.ndim == 0:
#         strides = ()
#     elif arrtype.layout == "C":
#         strides = [itemsize]
#         for dimension_size in reversed(shapes[1:]):
#             strides.append(builder.mul(strides[-1], dimension_size))
#         strides = tuple(reversed(strides))
#     elif arrtype.layout == "F":
#         strides = [itemsize]
#         for dimension_size in shapes[:-1]:
#             strides.append(builder.mul(strides[-1], dimension_size))
#         strides = tuple(strides)
#     else:
#         raise NotImplementedError(
#             "Don't know how to allocate array with layout '{0}'.".format(
#                 arrtype.layout
#             )
#         )

#     # Check overflow, numpy also does this after checking order
#     allocsize_mult = builder.smul_with_overflow(arrlen, itemsize)
#     allocsize = builder.extract_value(allocsize_mult, 0)
#     overflow = builder.or_(overflow, builder.extract_value(allocsize_mult, 1))

#     with builder.if_then(overflow, likely=False):
#         # Raise same error as numpy, see:
#         # https://github.com/numpy/numpy/blob/2a488fe76a0f732dc418d03b452caace161673da/numpy/core/src/multiarray/ctors.c#L1095-L1101    # noqa: E501
#         context.call_conv.return_user_exc(
#             builder,
#             ValueError,
#             (
#                 "array is too big; `arr.size * arr.dtype.itemsize` is larger "
#                 "than the maximum possible size.",
#             ),
#         )

#     usm_ty = arrtype.usm_type
#     usm_ty_val = 0
#     if usm_ty == "device":
#         usm_ty_val = 1
#     elif usm_ty == "shared":
#         usm_ty_val = 2
#     elif usm_ty == "host":
#         usm_ty_val = 3
#     usm_type = context.get_constant(types.uint64, usm_ty_val)
#     device = context.insert_const_string(builder.module, arrtype.device)

#     args = (
#         context.get_dummy_value(),
#         allocsize,
#         usm_type,
#         device,
#     )
#     mip = types.MemInfoPointer(types.voidptr)
#     arytypeclass = types.TypeRef(type(arrtype))
#     sig = signature(
#         mip,
#         arytypeclass,
#         types.intp,
#         types.uint64,
#         types.voidptr,
#     )

#     op = _call_usm_allocator
#     fnop = context.typing_context.resolve_value_type(op)
#     # The _call_usm_allocator function will be compiled and added to registry
#     # when the get_call_type function is invoked.
#     fnop.get_call_type(context.typing_context, sig.args, {})
#     eqfn = context.get_function(fnop, sig)
#     meminfo = eqfn(builder, args)
#     data = context.nrt.meminfo_data(builder, meminfo)
#     intp_t = context.get_value_type(types.intp)
#     shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
#     strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

#     populate_array(
#         ary,
#         data=builder.bitcast(data, datatype.as_pointer()),
#         shape=shape_array,
#         strides=strides_array,
#         itemsize=itemsize,
#         meminfo=meminfo,
#     )

#     return ary


def alloc_empty_arrayobj(context, builder, sig, llargs, is_like=False):
    """Construct an empty numba.np.arrayobj.make_array.<locals>.ArrayStruct

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        sig (numba.core.typing.templates.Signature): A numba's function
            signature object.
        llargs (tuple): A tuple of args to be parsed as the arguments of
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
        _parse_empty_like_args(context, builder, sig, llargs)
        if is_like
        else _parse_empty_args(context, builder, sig, llargs)
    )
    ary = _empty_nd_impl(context, builder, *arrtype)

    return ary, arrtype


def fill_arrayobj(context, builder, sig, llargs, value, is_like=False):
    """Fill a numba.np.arrayobj.make_array.<locals>.ArrayStruct
        with a specified value.

    Args:
        context (numba.core.base.BaseContext): One of the class derived
            from numba's BaseContext, e.g. CPUContext
        builder (llvmlite.ir.builder.IRBuilder): IR builder object from
            llvmlite.
        sig (numba.core.typing.templates.Signature): A numba's function
            signature object.
        llargs (tuple): A tuple of args to be parsed as the arguments of
            an np.empty(), np.zeros() or np.ones() call.
        value (int): The value to be set.
        is_like (bool, optional): Decides on how to parse the args.
            Defaults to False.

    Returns:
        tuple(numba.np.arrayobj.make_array.<locals>.ArrayStruct,
            numba_dpex.core.types.dpnp_ndarray_type.DpnpNdArray):
                A tuple of allocated array and constructed array type info
                in DpnpNdArray.
    """

    ary, arrtype = alloc_empty_arrayobj(context, builder, sig, llargs, is_like)
    itemsize = context.get_constant(
        types.intp, get_itemsize(context, arrtype[0])
    )
    device = context.insert_const_string(builder.module, arrtype[0].device)
    value = context.get_constant(types.int8, value)
    if isinstance(arrtype[0].dtype, types.scalars.Float):
        is_float = context.get_constant(types.boolean, 1)
    else:
        is_float = context.get_constant(types.boolean, 0)
    dpexrtCtx = dpexrt.DpexRTContext(context)
    dpexrtCtx.meminfo_fill(
        builder, ary.meminfo, itemsize, is_float, value, device
    )
    return ary, arrtype


@intrinsic
def intrin_usm_alloc(typingctx, allocsize, usm_type, device):
    """Intrinsic to call into the allocator for Array"""

    def codegen(context, builder, signature, args):
        [allocsize, usm_type, device] = args
        dpexrtCtx = dpexrt.DpexRTContext(context)
        meminfo = dpexrtCtx.meminfo_alloc(builder, allocsize, usm_type, device)
        return meminfo

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, usm_type, device)
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
        ty_shape (numba.core.types.abstract): One of the numba defined
            abstract types.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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

    def codegen(context, builder, sig, llargs):
        ary, _ = alloc_empty_arrayobj(context, builder, sig, llargs)
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
        ty_shape (numba.core.types.abstract): One of the numba defined
            abstract types.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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

    def codegen(context, builder, sig, llargs):
        ary, _ = fill_arrayobj(context, builder, sig, llargs, 0)
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
        ty_shape (numba.core.types.abstract): One of the numba defined
            abstract types.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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

    def codegen(context, builder, sig, llargs):
        ary, _ = fill_arrayobj(context, builder, sig, llargs, 1)
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_empty_like(
    ty_context,
    ty_x,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.empty_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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
        ty_x,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, llargs):
        ary, _ = alloc_empty_arrayobj(
            context, builder, sig, llargs, is_like=True
        )
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_zeros_like(
    ty_context,
    ty_x,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.zeros_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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
        ty_x,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, llargs):
        ary, _ = fill_arrayobj(context, builder, sig, llargs, 0, is_like=True)
        return ary._getvalue()

    return sig, codegen


@intrinsic
def impl_dpnp_ones_like(
    ty_context,
    ty_x,
    ty_dtype,
    ty_order,
    ty_device,
    ty_usm_type,
    ty_sycl_queue,
    ty_retty_ref,
):
    """A numba "intrinsic" function to inject code for dpnp.ones_like().

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.
        ty_x (numba.core.types.npytypes.Array): Numba type class for ndarray.
        ty_dtype (numba.core.types.functions.NumberClass): Type class for
            number classes (e.g. "np.float64").
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
        ty_x,
        ty_dtype,
        ty_order,
        ty_device,
        ty_usm_type,
        ty_sycl_queue,
        ty_retty_ref,
    )

    def codegen(context, builder, sig, llargs):
        ary, _ = fill_arrayobj(context, builder, sig, llargs, 1, is_like=True)
        return ary._getvalue()

    return sig, codegen
