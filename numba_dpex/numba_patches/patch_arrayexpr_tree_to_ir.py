# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import copy
import math
import operator

import dpnp
from numba.core import errors, ir, types, typing
from numba.core.ir_utils import mk_unique_var
from numba.core.typing import npydecl
from numba.parfors import array_analysis, parfor

from numba_dpex.core.typing import dpnpdecl


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


def get_dpnp_ufunc_typ(func):
    """get type of the incoming function from builtin registry"""
    for k, v in dpnpdecl.registry.globals:
        if k == func:
            return v
    raise RuntimeError("type for func ", func, " not found")


def _gen_dpnp_divide(arg1, arg2, out_ir, typemap):
    """generate np.divide() instead of / for array_expr to get numpy error model
    like inf for division by zero (test_division_by_zero).
    """
    scope = arg1.scope
    loc = arg1.loc
    g_np_var = ir.Var(scope, mk_unique_var("$np_g_var"), loc)
    typemap[g_np_var.name] = types.misc.Module(dpnp)
    g_np = ir.Global("dpnp", dpnp, loc)
    g_np_assign = ir.Assign(g_np, g_np_var, loc)
    # attr call: div_attr = getattr(g_np_var, divide)
    div_attr_call = ir.Expr.getattr(g_np_var, "divide", loc)
    attr_var = ir.Var(scope, mk_unique_var("$div_attr"), loc)
    func_var_typ = get_dpnp_ufunc_typ(dpnp.divide)
    typemap[attr_var.name] = func_var_typ
    attr_assign = ir.Assign(div_attr_call, attr_var, loc)
    # divide call:  div_attr(arg1, arg2)
    div_call = ir.Expr.call(attr_var, [arg1, arg2], (), loc)
    func_typ = func_var_typ.get_call_type(
        typing.Context(), [typemap[arg1.name], typemap[arg2.name]], {}
    )
    out_ir.extend([g_np_assign, attr_assign])
    return func_typ, div_call


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
                    # NUMBA_DPEX: is_dpnp_func check was added
                    func_typ, ir_expr = _gen_dpnp_divide(
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
                func_var_name = parfor._find_func_var(
                    typemap, op, avail_vars, loc=loc
                )
                func_var = ir.Var(scope, mk_unique_var(func_var_name), loc)
                typemap[func_var.name] = typemap[func_var_name]
                func_var_def = copy.deepcopy(
                    func_ir.get_definition(func_var_name)
                )
                if (
                    isinstance(func_var_def, ir.Expr)
                    and func_var_def.op == "getattr"
                    and func_var_def.attr == "sqrt"
                ):
                    g_math_var = ir.Var(
                        scope, mk_unique_var("$math_g_var"), loc
                    )
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
        # NUMBA_DPEX: is_dpnp_func check was added
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
