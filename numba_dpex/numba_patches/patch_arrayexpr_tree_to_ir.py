# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import copy
import math
import operator

from numba.core import errors, ir, types, typing
from numba.core.ir_utils import mk_unique_var
from numba.core.typing import npydecl
from numba.parfors import array_analysis, parfor


def _ufunc_to_parfor_instr(
    typemap,
    op,
    avail_vars,
    loc,
    scope,
    func_ir,
    out_ir,
    arg_vars,
    typingctx,
    calltypes,
    expr_out_var,
):
    func_var_name = parfor._find_func_var(typemap, op, avail_vars, loc=loc)
    func_var = ir.Var(scope, mk_unique_var(func_var_name), loc)
    typemap[func_var.name] = typemap[func_var_name]
    func_var_def = copy.deepcopy(func_ir.get_definition(func_var_name))
    if (
        isinstance(func_var_def, ir.Expr)
        and func_var_def.op == "getattr"
        and func_var_def.attr == "sqrt"
    ):
        g_math_var = ir.Var(scope, mk_unique_var("$math_g_var"), loc)
        typemap[g_math_var.name] = types.misc.Module(math)
        g_math = ir.Global("math", math, loc)
        g_math_assign = ir.Assign(g_math, g_math_var, loc)
        func_var_def = ir.Expr.getattr(g_math_var, "sqrt", loc)
        out_ir.append(g_math_assign)
    ir_expr = ir.Expr.call(func_var, arg_vars, (), loc)
    call_typ = typemap[func_var.name].get_call_type(
        typingctx, tuple(typemap[a.name] for a in arg_vars), {}
    )
    calltypes[ir_expr] = call_typ
    el_typ = call_typ.return_type
    out_ir.append(ir.Assign(func_var_def, func_var, loc))
    out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))

    return el_typ


def _arrayexpr_tree_to_ir(
    func_ir,
    typingctx,
    typemap,
    calltypes,
    equiv_set,
    init_block,
    expr_out_var,
    expr,
    parfor_index_tuple_var,
    all_parfor_indices,
    avail_vars,
):
    """generate IR from array_expr's expr tree recursively. Assign output to
    expr_out_var and returns the whole IR as a list of Assign nodes.
    """
    el_typ = typemap[expr_out_var.name]
    scope = expr_out_var.scope
    loc = expr_out_var.loc
    out_ir = []

    if isinstance(expr, tuple):
        op, arr_expr_args = expr
        arg_vars = []
        for arg in arr_expr_args:
            arg_out_var = ir.Var(scope, mk_unique_var("$arg_out_var"), loc)
            typemap[arg_out_var.name] = el_typ
            out_ir += _arrayexpr_tree_to_ir(
                func_ir,
                typingctx,
                typemap,
                calltypes,
                equiv_set,
                init_block,
                arg_out_var,
                arg,
                parfor_index_tuple_var,
                all_parfor_indices,
                avail_vars,
            )
            arg_vars.append(arg_out_var)
        if op in npydecl.supported_array_operators:
            el_typ1 = typemap[arg_vars[0].name]
            if len(arg_vars) == 2:
                el_typ2 = typemap[arg_vars[1].name]
                func_typ = typingctx.resolve_function_type(
                    op, (el_typ1, el_typ2), {}
                )
                ir_expr = ir.Expr.binop(op, arg_vars[0], arg_vars[1], loc)
                if op == operator.truediv:
                    func_typ, ir_expr = parfor._gen_np_divide(
                        arg_vars[0], arg_vars[1], out_ir, typemap
                    )
            else:
                func_typ = typingctx.resolve_function_type(op, (el_typ1,), {})
                ir_expr = ir.Expr.unary(op, arg_vars[0], loc)
            calltypes[ir_expr] = func_typ
            el_typ = func_typ.return_type
            out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
        for T in array_analysis.MAP_TYPES:
            if isinstance(op, T):
                # function calls are stored in variables which are not removed
                # op is typing_key to the variables type
                el_typ = _ufunc_to_parfor_instr(
                    typemap,
                    op,
                    avail_vars,
                    loc,
                    scope,
                    func_ir,
                    out_ir,
                    arg_vars,
                    typingctx,
                    calltypes,
                    expr_out_var,
                )
        if hasattr(op, "is_dpnp_ufunc"):
            el_typ = _ufunc_to_parfor_instr(
                typemap,
                op,
                avail_vars,
                loc,
                scope,
                func_ir,
                out_ir,
                arg_vars,
                typingctx,
                calltypes,
                expr_out_var,
            )
    elif isinstance(expr, ir.Var):
        var_typ = typemap[expr.name]
        if isinstance(var_typ, types.Array):
            el_typ = var_typ.dtype
            ir_expr = parfor._gen_arrayexpr_getitem(
                equiv_set,
                expr,
                parfor_index_tuple_var,
                all_parfor_indices,
                el_typ,
                calltypes,
                typingctx,
                typemap,
                init_block,
                out_ir,
            )
        else:
            el_typ = var_typ
            ir_expr = expr
        out_ir.append(ir.Assign(ir_expr, expr_out_var, loc))
    elif isinstance(expr, ir.Const):
        el_typ = typing.Context().resolve_value_type(expr.value)
        out_ir.append(ir.Assign(expr, expr_out_var, loc))

    if len(out_ir) == 0:
        raise errors.UnsupportedRewriteError(
            f"Don't know how to translate array expression '{expr:r}'",
            loc=expr.loc,
        )
    typemap.pop(expr_out_var.name, None)
    typemap[expr_out_var.name] = el_typ
    return out_ir


def patch():
    """
    Patches the _arrayexpr_tree_to_ir function in numba.parfor.parfor.py to
    support array expression nodes that were generated from dpnp expressions.
    """

    parfor._arrayexpr_tree_to_ir = _arrayexpr_tree_to_ir
