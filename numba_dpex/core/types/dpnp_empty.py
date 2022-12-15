# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
import numba.np.arrayobj
from llvmlite import ir
from numba import types
from numba.core import cgutils
from numba.extending import (
    intrinsic,
    lower_builtin,
    overload_classmethod,
    type_callable,
)

from .dpnp_types import dpnp_ndarray_Type


@type_callable(dpnp.empty)
def type_dpnp_empty(context):
    def typer(shape, dtype=None, usm_type=None, sycl_queue=None):
        from numba.core.typing.npydecl import parse_dtype, parse_shape

        if dtype is None:
            nb_dtype = types.double
        else:
            nb_dtype = parse_dtype(dtype)

        ndim = parse_shape(shape)

        if usm_type is None:
            usm_type = "device"
        else:
            usm_type = parse_usm_type(usm_type)

        if sycl_queue is None:
            sycl_queue = "0"

        if nb_dtype is not None and ndim is not None and usm_type is not None:
            return dpnp_ndarray_Type(
                dtype=nb_dtype,
                ndim=ndim,
                layout="C",
                usm_type=usm_type,
                sycl_queue=sycl_queue,
            )

    return typer


def parse_usm_type(usm_type):
    """
    Return the usm_type, if it is a string literal.
    """
    from numba.core.errors import TypingError

    if isinstance(usm_type, types.StringLiteral):
        usm_type_str = usm_type.literal_value
        if usm_type_str not in ["shared", "device", "host"]:
            msg = f"Invalid usm_type specified: '{usm_type_str}'"
            raise TypingError(msg)
        return usm_type_str


@lower_builtin(dpnp.empty, types.Any, types.Any, types.Any, types.Any)
def impl_dpnp_empty(context, builder, sig, args):
    """
    Inputs: shape, dtype, usm_type, queue
    """
    from numba.core.imputils import impl_ret_new_ref

    empty_args = _parse_empty_args(context, builder, sig, args)
    ary = _empty_nd_impl(context, builder, *empty_args)
    return impl_ret_new_ref(context, builder, sig.return_type, ary._getvalue())


def _parse_empty_args(context, builder, sig, args):
    """
    Parse the arguments of a dpnp.empty(), .zeros() or .ones() call.
    """
    from numba.np.arrayobj import _parse_shape

    arrtype = sig.return_type

    arrshapetype = sig.args[0]
    arrshape = args[0]
    shape = _parse_shape(context, builder, arrshapetype, arrshape)

    queue = args[-1]
    return (arrtype, shape, queue)


def _empty_nd_impl(context, builder, arrtype, shapes):
    """See numba.np.arrayobj._empty_nd_impl().
    This implementation uses different MemInfo allocator.
    """
    if not isinstance(arrtype, dpnp_ndarray_Type):
        return tmpCopy(context, builder, arrtype, shapes)

    from numba.np.arrayobj import (
        get_itemsize,
        make_array,
        populate_array,
        signature,
    )

    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    overflow = ir.Constant(ir.IntType(1), 0)
    for s in shapes:
        arrlen_mult = builder.smul_with_overflow(arrlen, s)
        arrlen = builder.extract_value(arrlen_mult, 0)
        overflow = builder.or_(overflow, builder.extract_value(arrlen_mult, 1))

    if arrtype.ndim == 0:
        strides = ()
    elif arrtype.layout == "C":
        strides = [itemsize]
        for dimension_size in reversed(shapes[1:]):
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(reversed(strides))
    elif arrtype.layout == "F":
        strides = [itemsize]
        for dimension_size in shapes[:-1]:
            strides.append(builder.mul(strides[-1], dimension_size))
        strides = tuple(strides)
    else:
        raise NotImplementedError(
            "Don't know how to allocate array with layout '{0}'.".format(
                arrtype.layout
            )
        )

    # Check overflow, numpy also does this after checking order
    allocsize_mult = builder.smul_with_overflow(arrlen, itemsize)
    allocsize = builder.extract_value(allocsize_mult, 0)
    overflow = builder.or_(overflow, builder.extract_value(allocsize_mult, 1))

    with builder.if_then(overflow, likely=False):
        # Raise same error as numpy, see:
        # https://github.com/numpy/numpy/blob/2a488fe76a0f732dc418d03b452caace161673da/numpy/core/src/multiarray/ctors.c#L1095-L1101    # noqa: E501
        context.call_conv.return_user_exc(
            builder,
            ValueError,
            (
                "array is too big; `arr.size * arr.dtype.itemsize` is larger than"
                " the maximum possible size.",
            ),
        )

    usm_type_num = {"shared": 0, "device": 1, "host": 2}[arrtype.usm_type]
    usm_type = context.get_constant(types.int64, usm_type_num)
    sycl_queue_type = context.get_constant(types.voidptr, arrtype.sycl_queue)

    args = (context.get_dummy_value(), allocsize, usm_type, sycl_queue_type)
    mip = types.MemInfoPointer(types.voidptr)
    arytypeclass = types.TypeRef(type(arrtype))
    sig = signature(mip, arytypeclass, types.intp, types.intp, types.voidptr)

    meminfo = context.compile_internal(builder, _call_allocator, sig, args)
    data = context.nrt.meminfo_data(builder, meminfo)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, shapes, ty=intp_t)
    strides_array = cgutils.pack_array(builder, strides, ty=intp_t)

    populate_array(
        ary,
        data=builder.bitcast(data, datatype.as_pointer()),
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=meminfo,
    )

    return ary


tmpCopy = numba.np.arrayobj._empty_nd_impl
numba.np.arrayobj._empty_nd_impl = _empty_nd_impl


def _call_allocator(arrtype, size, usm_type, sycl_queue):
    """Trampoline to call the intrinsic used for allocation"""
    return arrtype._allocate(size, usm_type, sycl_queue)


@overload_classmethod(dpnp_ndarray_Type, "_allocate")
def _ol_dpnp_array_allocate(cls, size, usm_type, sycl_queue):
    def impl(cls, size, usm_type, sycl_queue):
        return intrin_alloc(size, usm_type, sycl_queue)

    return impl


@intrinsic
def intrin_alloc(typingctx, size, usm_type, sycl_queue):
    """Intrinsic to call into the allocator for Array"""
    from numba.core.base import BaseContext
    from numba.core.runtime.context import NRTContext
    from numba.core.typing.templates import Signature

    def MemInfo_new(context: NRTContext, builder, size, usm_type, queue):
        context._require_nrt()

        mod = builder.module
        fnargs = [cgutils.intp_t, cgutils.intp_t, cgutils.voidptr_t]
        fnty = ir.FunctionType(cgutils.voidptr_t, fnargs)
        fn = cgutils.get_or_insert_function(mod, fnty, "DPRT_MemInfo_new")
        fn.return_value.add_attribute("noalias")
        return builder.call(fn, [size, usm_type, queue])

    def codegen(context: BaseContext, builder, signature: Signature, args):
        meminfo = MemInfo_new(context.nrt, builder, *args)
        meminfo.name = "meminfo"
        return meminfo

    from numba.core.typing import signature

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, size, usm_type, sycl_queue)
    return sig, codegen
