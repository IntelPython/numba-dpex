# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import dpnp
from numba.core import ir, types
from numba.core.ir_utils import get_np_ufunc_typ, mk_unique_var

from .usm_ndarray_type import USMNdArray


def partialclass(cls, *args, **kwds):
    """Creates fabric class of the original class with preset initialization
    arguments."""
    cls0 = partial(cls, *args, **kwds)
    new_cls = type(
        cls.__name__ + "Partial",
        (cls,),
        {"__new__": lambda cls, *args, **kwds: cls0(*args, **kwds)},
    )

    return new_cls


class DpnpNdArray(USMNdArray):
    """
    The Numba type to represent an dpnp.ndarray. The type has the same
    structure as USMNdArray used to represent dpctl.tensor.usm_ndarray.
    """

    @property
    def is_internal(self):
        """Sets the property so that DpnpNdArray expressions can be converted
        to Numba array_expression objects.

        Returns:
            bool: Always returns True.
        """
        return True

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Used to derive the output of array operations involving DpnpNdArray
        type objects.

        Note that the __array_ufunc__ returns the LHS value of an expression
        involving DpnpNdArray. However, the resulting type is not complete and
        is default initialized. As such, the LHS type does not meet compute
        follows data and usm_type propagation rules.

        The ParforLegalizeCFDPass fixes the typing of the LHS array expression.

        Returns: The DpnpNdArray class.
        """
        if method == "__call__":
            dpnp_type = None

            for inp in inputs:
                if isinstance(inp, DpnpNdArray):
                    dpnp_type = inp
                    continue
                if isinstance(inp, types.abstract.Number):
                    continue

                return NotImplemented

            assert dpnp_type is not None

            return partialclass(
                DpnpNdArray, queue=dpnp_type.queue, usm_type=dpnp_type.usm_type
            )
        else:
            return

    def __str__(self):
        return self.name.replace("USMNdArray", "DpnpNdArray")

    def __repr__(self):
        return self.__str__()

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
        # dpex specific argument:
        queue_ir_val=None,
    ):
        """Generates the Numba typed IR representing the allocation of a new
        DpnpNdArray using the dpnp.ndarray overload.

        Args:
            typingctx: The typing context used when generating the IR.
            typemap: The current IR state's typemap
            calltypes: The calltype of the dpnp.empty function
            lhs: The LHS IR value for the result of __allocate__
            size_var: The size of the allocation
            dtype: The dtype of the array to be allocated.
            scope: The scope in which the allocated array is alive
            loc: Location in the source
            lhs_typ: The dtype of the LHS value of the allocate call.
            size_typ: The dtype of the size Value
            out: The output array.

        Raises:
            NotImplementedError: Thrown if the allocation is for a F-contiguous
            array

        Returns: The IR Value for the allocated array
        """
        # TODO: it looks like it is being called only for parfor allocations,
        # so can we rely on it? We can grab information from input arguments
        # from rhs, but doc does not set any restriction on parfor use only.
        assert queue_ir_val is not None
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
        typ_var = ir.Var(scope, mk_unique_var("$np_dtype_var"), loc)
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

        # A default usm_type arg added as a placeholder
        layout_var = ir.Var(scope, mk_unique_var("$layout_var"), loc)
        usm_typ_var = ir.Var(scope, mk_unique_var("$np_usm_type_var"), loc)
        # A default device string arg added as a placeholder
        device_typ_var = ir.Var(scope, mk_unique_var("$np_device_var"), loc)
        queue_typ_var = ir.Var(scope, mk_unique_var("$np_queue_var"), loc)

        if typemap:
            typemap[layout_var.name] = types.literal(lhs_typ.layout)
            typemap[usm_typ_var.name] = types.literal(lhs_typ.usm_type)
            typemap[device_typ_var.name] = types.none
            typemap[queue_typ_var.name] = lhs_typ.queue

        layout_var_assign = ir.Assign(
            ir.Const(lhs_typ.layout, loc), layout_var, loc
        )
        usm_typ_var_assign = ir.Assign(
            ir.Const(lhs_typ.usm_type, loc), usm_typ_var, loc
        )
        device_typ_var_assign = ir.Assign(
            ir.Const(None, loc), device_typ_var, loc
        )
        queue_typ_var_assign = ir.Assign(queue_ir_val, queue_typ_var, loc)

        out.extend(
            [
                layout_var_assign,
                usm_typ_var_assign,
                device_typ_var_assign,
                queue_typ_var_assign,
            ]
        )

        alloc_call = ir.Expr.call(
            attr_var,
            [
                size_var,
                typ_var,
                layout_var,
                device_typ_var,
                usm_typ_var,
                queue_typ_var,
            ],
            (),
            loc,
        )

        if calltypes:
            cac = typemap[attr_var.name].get_call_type(
                typingctx,
                [
                    typemap[x.name]
                    for x in [
                        size_var,
                        typ_var,
                        layout_var,
                        device_typ_var,
                        usm_typ_var,
                        queue_typ_var,
                    ]
                ],
                {},
            )
            # By default, all calls to "empty" are typed as returning a standard
            # NumPy ndarray.  If we are allocating a ndarray subclass here then
            # just change the return type to be that of the subclass.
            cac._return_type = (
                lhs_typ.copy(layout="C") if lhs_typ.layout == "F" else lhs_typ
            )
            calltypes[alloc_call] = cac

        if lhs_typ.layout == "F":
            # The F contiguous layout needs to be properly tested before we
            # enable this code path.
            raise NotImplementedError("F Arrays are not yet supported")

        else:
            alloc_assign = ir.Assign(alloc_call, lhs, loc)
            out.extend([g_np_assign, attr_assign, typ_var_assign, alloc_assign])

        return out
