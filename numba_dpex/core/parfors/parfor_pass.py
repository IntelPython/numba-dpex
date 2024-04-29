# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
This module follows the logic of numba/parfors/parfor.py with changes required
to use it with dpnp instead of numpy.
"""


import copy
import math
import operator
import warnings

import dpnp
from numba.core import config, errors, ir, types, typing
from numba.core.compiler_machinery import register_pass
from numba.core.ir_utils import (
    convert_size_to_var,
    dprint_func_ir,
    mk_unique_var,
    next_label,
)
from numba.core.typed_passes import ParforPass as NumpyParforPass
from numba.core.typed_passes import _reload_parfors
from numba.core.typing import npydecl
from numba.parfors import array_analysis, parfor
from numba.parfors.parfor import (
    ConvertInplaceBinop,
    ConvertLoopPass,
    ConvertNumpyPass,
    ConvertReducePass,
    ConvertSetItemPass,
    Parfor,
)
from numba.parfors.parfor import ParforPass as _NumpyParforPass
from numba.parfors.parfor import (
    _make_index_var,
    _mk_parfor_loops,
    repr_arrayexpr,
    signature,
)
from numba.stencils.stencilparfor import StencilPass

from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray
from numba_dpex.core.typing import dpnpdecl


class ConvertDPNPPass(ConvertNumpyPass):
    """
    Convert supported Dpnp functions, as well as arrayexpr nodes, to
    parfor nodes.

    Based on the ConvertNumpyPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    def __init__(self, pass_states):
        super().__init__(pass_states)

    def _get_queue(self, queue_type, expr: tuple):
        """
        Extracts queue from the input arguments of the array operation.
        """
        pass_states = self.pass_states
        typemap: map[str, any] = pass_states.typemap

        var_with_queue = None

        for var in expr[1]:
            if isinstance(var, tuple):
                res = self._get_queue(queue_type, var)
                if res is not None:
                    return res

                continue

            if not isinstance(var, ir.Var):
                continue

            _type = typemap[var.name]
            if not isinstance(_type, DpnpNdArray):
                continue
            if queue_type != _type.queue:
                continue

            var_with_queue = var
            break

        return ir.Expr.getattr(var_with_queue, "sycl_queue", var_with_queue.loc)

    def _arrayexpr_to_parfor(self, equiv_set, lhs, arrayexpr, avail_vars):
        """generate parfor from arrayexpr node, which is essentially a
        map with recursive tree.

        Exactly same as the original one, but with mock to _arrayexpr_tree_to_ir
        """
        pass_states = self.pass_states
        scope = lhs.scope
        loc = lhs.loc
        expr = arrayexpr.expr
        arr_typ = pass_states.typemap[lhs.name]
        el_typ = arr_typ.dtype

        # generate loopnests and size variables from lhs correlations
        size_vars = equiv_set.get_shape(lhs)
        index_vars, loopnests = _mk_parfor_loops(
            pass_states.typemap, size_vars, scope, loc
        )

        # Expr is a tuple
        ir_queue = self._get_queue(arr_typ.queue, expr)
        assert ir_queue is not None

        # generate init block and body
        init_block = ir.Block(scope, loc)
        init_block.body = mk_alloc(
            pass_states.typingctx,
            pass_states.typemap,
            pass_states.calltypes,
            lhs,
            tuple(size_vars),
            el_typ,
            scope,
            loc,
            pass_states.typemap[lhs.name],
            queue_ir_val=ir_queue,
        )
        body_label = next_label()
        body_block = ir.Block(scope, loc)
        expr_out_var = ir.Var(scope, mk_unique_var("$expr_out_var"), loc)
        pass_states.typemap[expr_out_var.name] = el_typ

        index_var, index_var_typ = _make_index_var(
            pass_states.typemap, scope, index_vars, body_block
        )

        body_block.body.extend(
            _arrayexpr_tree_to_ir(
                pass_states.func_ir,
                pass_states.typingctx,
                pass_states.typemap,
                pass_states.calltypes,
                equiv_set,
                init_block,
                expr_out_var,
                expr,
                index_var,
                index_vars,
                avail_vars,
            )
        )

        pat = ("array expression {}".format(repr_arrayexpr(arrayexpr.expr)),)

        parfor = Parfor(
            loopnests,
            init_block,
            {},
            loc,
            index_var,
            equiv_set,
            pat[0],
            pass_states.flags,
        )

        setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
        pass_states.calltypes[setitem_node] = signature(
            types.none, pass_states.typemap[lhs.name], index_var_typ, el_typ
        )
        body_block.body.append(setitem_node)
        parfor.loop_body = {body_label: body_block}
        if config.DEBUG_ARRAY_OPT >= 1:
            print("parfor from arrayexpr")
            parfor.dump()
        return parfor


class _ParforPass(_NumpyParforPass):
    """ParforPass class is responsible for converting NumPy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.

    Based on the _NumpyParforPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR.

        Exactly same as the original one, but with mock ConvertNumpyPass to
        ConvertDPNPPass.
        """
        self._pre_run()
        # run stencil translation to parfor
        if self.options.stencil:
            stencil_pass = StencilPass(
                self.func_ir,
                self.typemap,
                self.calltypes,
                self.array_analysis,
                self.typingctx,
                self.targetctx,
                self.flags,
            )
            stencil_pass.run()
        if self.options.setitem:
            ConvertSetItemPass(self).run(self.func_ir.blocks)
        if self.options.numpy:
            ConvertDPNPPass(self).run(self.func_ir.blocks)
        if self.options.reduction:
            ConvertReducePass(self).run(self.func_ir.blocks)
        if self.options.prange:
            ConvertLoopPass(self).run(self.func_ir.blocks)
        if self.options.inplace_binop:
            ConvertInplaceBinop(self).run(self.func_ir.blocks)

        # setup diagnostics now parfors are found
        self.diagnostics.setup(self.func_ir, self.options.fusion)

        dprint_func_ir(self.func_ir, "after parfor pass")


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPass(NumpyParforPass):
    """Based on the NumpyParforPass. Lot's of code was copy-pasted, with minor
    changes due to lack of extensibility of the original package.
    """

    _name = "dpnp_parfor_pass"

    def __init__(self):
        NumpyParforPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes.

        Exactly same as the original one, but with mock to _ParforPass.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _ParforPass(
            state.func_ir,
            state.typemap,
            state.calltypes,
            state.return_type,
            state.typingctx,
            state.targetctx,
            state.flags.auto_parallel,
            state.flags,
            state.metadata,
            state.parfor_diagnostics,
        )
        parfor_pass.run()

        # check the parfor pass worked and warn if it didn't
        has_parfor = False
        for blk in state.func_ir.blocks.values():
            for stmnt in blk.body:
                if isinstance(stmnt, Parfor):
                    has_parfor = True
                    break
            else:
                continue
            break

        if not has_parfor:
            # parfor calls the compiler chain again with a string
            if not (
                config.DISABLE_PERFORMANCE_WARNINGS
                or state.func_ir.loc.filename == "<string>"
            ):
                url = (
                    "https://numba.readthedocs.io/en/stable/user/"
                    "parallel.html#diagnostics"
                )
                msg = (
                    "\nThe keyword argument 'parallel=True' was specified "
                    "but no transformation for parallel execution was "
                    "possible.\n\nTo find out why, try turning on parallel "
                    "diagnostics, see %s for help." % url
                )
                warnings.warn(
                    errors.NumbaPerformanceWarning(msg, state.func_ir.loc)
                )

        # Add reload function to initialize the parallel backend.
        state.reload_init.append(_reload_parfors)
        return True


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


def mk_alloc(
    typingctx,
    typemap,
    calltypes,
    lhs,
    size_var,
    dtype,
    scope,
    loc,
    lhs_typ,
    **kws,
):
    """generate an array allocation with np.empty() and return list of nodes.
    size_var can be an int variable or tuple of int variables.
    lhs_typ is the type of the array being allocated.

    Taken from numba, added kws argument to pass it to __allocate__
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
            **kws,
        )

    # Unused numba's code..
    assert False
