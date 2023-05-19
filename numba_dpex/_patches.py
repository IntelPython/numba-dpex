# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from llvmlite.ir import Constant
from numba.core import cgutils
from numba.core import config as numba_config
from numba.core import types
from numba.core.typing import signature
from numba.extending import intrinsic, overload_classmethod
from numba.np.arrayobj import (
    _call_allocator,
    get_itemsize,
    make_array,
    populate_array,
)

from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpnpNdArray


def _empty_nd_impl(context, builder, arrtype, shapes):
    """Utility function used for allocating a new array during LLVM code
    generation (lowering).  Given a target context, builder, array
    type, and a tuple or list of lowered dimension sizes, returns a
    LLVM value pointing at a Numba runtime allocated array.
    """

    arycls = make_array(arrtype)
    ary = arycls(context, builder)

    datatype = context.get_data_type(arrtype.dtype)
    itemsize = context.get_constant(types.intp, get_itemsize(context, arrtype))

    # compute array length
    arrlen = context.get_constant(types.intp, 1)
    overflow = Constant(llvmir.IntType(1), 0)
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

    if isinstance(arrtype, DpnpNdArray):
        usm_ty = arrtype.usm_type
        usm_ty_val = 0
        if usm_ty == "device":
            usm_ty_val = 1
        elif usm_ty == "shared":
            usm_ty_val = 2
        elif usm_ty == "host":
            usm_ty_val = 3
        usm_type = context.get_constant(types.uint64, usm_ty_val)
        device = context.insert_const_string(builder.module, arrtype.device)

        args = (
            context.get_dummy_value(),
            allocsize,
            usm_type,
            device,
        )
        mip = types.MemInfoPointer(types.voidptr)
        arytypeclass = types.TypeRef(type(arrtype))
        sig = signature(
            mip,
            arytypeclass,
            types.intp,
            types.uint64,
            types.voidptr,
        )
        from numba_dpex.decorators import dpjit

        op = dpjit(_call_usm_allocator)
        fnop = context.typing_context.resolve_value_type(op)
        # The _call_usm_allocator function will be compiled and added to registry
        # when the get_call_type function is invoked.
        fnop.get_call_type(context.typing_context, sig.args, {})
        eqfn = context.get_function(fnop, sig)
        meminfo = eqfn(builder, args)
    else:
        dtype = arrtype.dtype
        align_val = context.get_preferred_array_alignment(dtype)
        align = context.get_constant(types.uint32, align_val)
        args = (context.get_dummy_value(), allocsize, align)

        mip = types.MemInfoPointer(types.voidptr)
        arytypeclass = types.TypeRef(type(arrtype))
        argtypes = signature(mip, arytypeclass, types.intp, types.uint32)

        meminfo = context.compile_internal(
            builder, _call_allocator, argtypes, args
        )

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


@overload_classmethod(DpnpNdArray, "_usm_allocate")
def _ol_array_allocate(cls, allocsize, usm_type, device):
    """Implements an allocator for dpnp.ndarrays."""

    def impl(cls, allocsize, usm_type, device):
        return intrin_usm_alloc(allocsize, usm_type, device)

    return impl


numba_config.DISABLE_PERFORMANCE_WARNINGS = 0


def _call_usm_allocator(arrtype, size, usm_type, device):
    """Trampoline to call the intrinsic used for allocation"""
    return arrtype._usm_allocate(size, usm_type, device)


numba_config.DISABLE_PERFORMANCE_WARNINGS = 1


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
