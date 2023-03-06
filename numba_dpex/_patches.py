# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numpy
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
from numba.core import cgutils
from numba.core import config as numba_config
from numba.core import ir, types
from numba.core.ir_utils import (
    convert_size_to_var,
    get_np_ufunc_typ,
    mk_unique_var,
)
from numba.core.typing import signature
from numba.extending import intrinsic, overload_classmethod
from numba.np.arrayobj import (
    _call_allocator,
    get_itemsize,
    make_array,
    populate_array,
)
from numba.np.ufunc.dufunc import DUFunc

from numba_dpex.core.runtime import context as dpexrt
from numba_dpex.core.types import DpnpNdArray

# Numpy array constructors


def _is_ufunc(func):
    return isinstance(func, (numpy.ufunc, DUFunc)) or hasattr(
        func, "is_dpnp_ufunc"
    )


def _mk_alloc(
    typingctx, typemap, calltypes, lhs, size_var, dtype, scope, loc, lhs_typ
):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    lhs_typ is the type of the array being allocated.
    """
    out = []
    ndims = 1
    size_typ = types.intp
    if isinstance(size_var, tuple):
        if len(size_var) == 1:
            size_var = size_var[0]
            size_var = convert_size_to_var(size_var, typemap, scope, loc, out)
        else:
            # tuple_var = build_tuple([size_var...])
            ndims = len(size_var)
            tuple_var = ir.Var(scope, mk_unique_var("$tuple_var"), loc)
            if typemap:
                typemap[tuple_var.name] = types.containers.UniTuple(
                    types.intp, ndims
                )
            # constant sizes need to be assigned to vars
            new_sizes = [
                convert_size_to_var(s, typemap, scope, loc, out)
                for s in size_var
            ]
            tuple_call = ir.Expr.build_tuple(new_sizes, loc)
            tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
            out.append(tuple_assign)
            size_var = tuple_var
            size_typ = types.containers.UniTuple(types.intp, ndims)

    if hasattr(lhs_typ, "__allocate__"):
        return lhs_typ.__allocate__(
            typingctx,
            typemap,
            calltypes,
            lhs,
            size_var,
            dtype,
            scope,
            loc,
            lhs_typ,
            size_typ,
            out,
        )

    # g_np_var = Global(numpy)
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    if typemap:
        typemap[g_np_var.name] = types.misc.Module(numpy)
    g_np = ir.Global("np", numpy, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: empty_attr = getattr(g_np_var, empty)
    empty_attr_call = ir.Expr.getattr(g_np_var, "empty", loc)
    attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
    if typemap:
        typemap[attr_var.name] = get_np_ufunc_typ(numpy.empty)
    attr_assign = ir.Assign(empty_attr_call, attr_var, loc)
    # Assume str(dtype) returns a valid type
    dtype_str = str(dtype)
    # alloc call: lhs = empty_attr(size_var, typ_var)
    typ_var = ir.Var(scope, mk_unique_var("$np_typ_var"), loc)
    if typemap:
        typemap[typ_var.name] = types.functions.NumberClass(dtype)
    # If dtype is a datetime/timedelta with a unit,
    # then it won't return a valid type and instead can be created
    # with a string. i.e. "datetime64[ns]")
    if (
        isinstance(dtype, (types.NPDatetime, types.NPTimedelta))
        and dtype.unit != ""
    ):
        typename_const = ir.Const(dtype_str, loc)
        typ_var_assign = ir.Assign(typename_const, typ_var, loc)
    else:
        if dtype_str == "bool":
            # empty doesn't like 'bool' sometimes (e.g. kmeans example)
            dtype_str = "bool_"
        np_typ_getattr = ir.Expr.getattr(g_np_var, dtype_str, loc)
        typ_var_assign = ir.Assign(np_typ_getattr, typ_var, loc)
    alloc_call = ir.Expr.call(attr_var, [size_var, typ_var], (), loc)

    if calltypes:
        cac = typemap[attr_var.name].get_call_type(
            typingctx, [size_typ, types.functions.NumberClass(dtype)], {}
        )
        # By default, all calls to "empty" are typed as returning a standard
        # NumPy ndarray.  If we are allocating a ndarray subclass here then
        # just change the return type to be that of the subclass.
        cac._return_type = (
            lhs_typ.copy(layout="C") if lhs_typ.layout == "F" else lhs_typ
        )
        calltypes[alloc_call] = cac
    if lhs_typ.layout == "F":
        empty_c_typ = lhs_typ.copy(layout="C")
        empty_c_var = ir.Var(scope, mk_unique_var("$empty_c_var"), loc)
        if typemap:
            typemap[empty_c_var.name] = lhs_typ.copy(layout="C")
        empty_c_assign = ir.Assign(alloc_call, empty_c_var, loc)

        # attr call: asfortranarray = getattr(g_np_var, asfortranarray)
        asfortranarray_attr_call = ir.Expr.getattr(
            g_np_var, "asfortranarray", loc
        )
        afa_attr_var = ir.Var(
            scope, mk_unique_var("$asfortran_array_attr"), loc
        )
        if typemap:
            typemap[afa_attr_var.name] = get_np_ufunc_typ(numpy.asfortranarray)
        afa_attr_assign = ir.Assign(asfortranarray_attr_call, afa_attr_var, loc)
        # call asfortranarray
        asfortranarray_call = ir.Expr.call(afa_attr_var, [empty_c_var], (), loc)
        if calltypes:
            calltypes[asfortranarray_call] = typemap[
                afa_attr_var.name
            ].get_call_type(typingctx, [empty_c_typ], {})

        asfortranarray_assign = ir.Assign(asfortranarray_call, lhs, loc)

        out.extend(
            [
                g_np_assign,
                attr_assign,
                typ_var_assign,
                empty_c_assign,
                afa_attr_assign,
                asfortranarray_assign,
            ]
        )
    else:
        alloc_assign = ir.Assign(alloc_call, lhs, loc)
        out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])

    return out


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
