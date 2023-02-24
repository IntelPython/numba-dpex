# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
from numba.core import cgutils, ir, types
from numba.core.errors import NumbaNotImplementedError
from numba.core.ir_utils import get_np_ufunc_typ, mk_unique_var
from numba.core.pythonapi import NativeValue, PythonAPI, box, unbox
from numba.np import numpy_support

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.runtime import context as dpexrt

from .usm_ndarray_type import USMNdArray


class DpnpNdArray(USMNdArray):
    """
    The Numba type to represent an dpnp.ndarray. The type has the same
    structure as USMNdArray used to represent dpctl.tensor.usm_ndarray.
    """

    @property
    def is_internal(self):
        return True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if not all(isinstance(inp, DpnpNdArray) for inp in inputs):
                return NotImplemented
            return DpnpNdArray
        else:
            return

    def __allocate__(
        self,
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
    ):
        # g_np_var = Global(dpnp)
        g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
        if typemap:
            typemap[g_np_var.name] = types.misc.Module(dpnp)
        g_np = ir.Global("np", dpnp, loc)
        g_np_assign = ir.Assign(g_np, g_np_var, loc)
        # attr call: empty_attr = getattr(g_np_var, empty)
        empty_attr_call = ir.Expr.getattr(g_np_var, "empty", loc)
        attr_var = ir.Var(scope, mk_unique_var("$empty_attr_attr"), loc)
        if typemap:
            typemap[attr_var.name] = get_np_ufunc_typ(dpnp.empty)
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
            assert False and "F Arrays are not yet supported"
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
                typemap[afa_attr_var.name] = get_np_ufunc_typ(
                    dpnp.asfortranarray
                )
            afa_attr_assign = ir.Assign(
                asfortranarray_attr_call, afa_attr_var, loc
            )
            # call asfortranarray
            asfortranarray_call = ir.Expr.call(
                afa_attr_var, [empty_c_var], (), loc
            )
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


# --------------- Boxing/Unboxing logic for dpnp.ndarray ----------------------#


@unbox(DpnpNdArray)
def unbox_dpnp_nd_array(typ, obj, c):
    """Converts a dpnp.ndarray object to a Numba internal array structure.

    Args:
        typ : The Numba type of the PyObject
        obj : The actual PyObject to be unboxed
        c :

    Returns:
        _type_: _description_
    """
    # Reusing the numba.core.base.BaseContext's make_array function to get a
    # struct allocated. The same struct is used for numpy.ndarray
    # and dpnp.ndarray. It is possible to do so, as the extra information
    # specific to dpnp.ndarray such as sycl_queue is inferred statically and
    # stored as part of the DpnpNdArray type.

    # --------------- Original Numba comment from @ubox(types.Array)
    #
    # This is necessary because unbox_buffer() does not work on some
    # dtypes, e.g. datetime64 and timedelta64.
    # TODO check matching dtype.
    #      currently, mismatching dtype will still work and causes
    #      potential memory corruption
    #
    # --------------- End of Numba comment from @ubox(types.Array)
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)
    aryptr = nativeary._getpointer()

    ptr = c.builder.bitcast(aryptr, c.pyapi.voidptr)
    # FIXME : We need to check if Numba_RT as well as DPEX RT are enabled.
    if c.context.enable_nrt:
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        errcode = dpexrtCtx.arraystruct_from_python(c.pyapi, obj, ptr)
    else:
        raise UnreachableError

    # TODO: here we have minimal typechecking by the itemsize.
    #       need to do better
    try:
        expected_itemsize = numpy_support.as_dtype(typ.dtype).itemsize
    except NumbaNotImplementedError:
        # Don't check types that can't be `as_dtype()`-ed
        itemsize_mismatch = cgutils.false_bit
    else:
        expected_itemsize = nativeary.itemsize.type(expected_itemsize)
        itemsize_mismatch = c.builder.icmp_unsigned(
            "!=",
            nativeary.itemsize,
            expected_itemsize,
        )

    failed = c.builder.or_(
        cgutils.is_not_null(c.builder, errcode),
        itemsize_mismatch,
    )
    # Handle error
    with c.builder.if_then(failed, likely=False):
        c.pyapi.err_set_string(
            "PyExc_TypeError",
            "can't unbox array from PyObject into "
            "native value.  The object maybe of a "
            "different type",
        )
    return NativeValue(c.builder.load(aryptr), is_error=failed)


@box(DpnpNdArray)
def box_array(typ, val, c):
    if c.context.enable_nrt:
        np_dtype = numpy_support.as_dtype(typ.dtype)
        dtypeptr = c.env_manager.read_const(c.env_manager.add_const(np_dtype))
        dpexrtCtx = dpexrt.DpexRTContext(c.context)
        newary = dpexrtCtx.usm_ndarray_to_python_acqref(
            c.pyapi, typ, val, dtypeptr
        )

        if not newary:
            c.pyapi.err_set_string(
                "PyExc_TypeError",
                "could not box native array into a dpnp.ndarray PyObject.",
            )

        # Steals NRT ref
        # Refer:
        #   numba.core.base.nrt -> numba.core.runtime.context -> decref
        #   The `NRT_decref` function is generated directly as LLVM IR inside
        #   numba.core.runtime.nrtdynmod.py
        c.context.nrt.decref(c.builder, typ, val)

        return newary
    else:
        raise UnreachableError
