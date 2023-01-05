# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import ir, types
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (
    find_topo_order,
    get_definition,
    mk_unique_var,
    remove_dead,
    simplify_CFG,
)

import numba_dpex

rewrite_function_name_map = {
    # numpy
    "all": (["numpy"], "all"),
    "amax": (["numpy"], "amax"),
    "amin": (["numpy"], "amin"),
    "argmax": (["numpy"], "argmax"),
    "argmin": (["numpy"], "argmin"),
    "argsort": (["numpy"], "argsort"),
    "cov": (["numpy"], "cov"),
    "diagonal": (["numpy"], "diagonal"),
    "max": (["numpy"], "max"),
    "mean": (["numpy"], "mean"),
    "median": (["numpy"], "median"),
    "min": (["numpy"], "min"),
    "partition": (["numpy"], "partition"),
    "repeat": (["numpy"], "repeat"),
    "trace": (["numpy"], "trace"),
    "vdot": (["numpy"], "vdot"),
    # random
    "beta": (["random"], "beta"),
    "binomial": (["random"], "binomial"),
    "chisquare": (["random"], "chisquare"),
    "exponential": (["random"], "exponential"),
    "gamma": (["random"], "gamma"),
    "geometric": (["random"], "geometric"),
    "gumbel": (["random"], "gumbel"),
    "hypergeometric": (["random"], "hypergeometric"),
    "laplace": (["random"], "laplace"),
    "lognormal": (["random"], "lognormal"),
    "multinomial": (["random"], "multinomial"),
    "multivariate_normal": (["random"], "multivariate_normal"),
    "negative_binomial": (["random"], "negative_binomial"),
    "normal": (["random"], "normal"),
    "poisson": (["random"], "poisson"),
    "rand": (["random"], "rand"),
    "randint": (["random"], "randint"),
    "random_integers": (["random"], "random_integers"),
    "random_sample": (["random"], "random_sample"),
    "random": (["random"], "random"),
    "ranf": (["random"], "ranf"),
    "rayleigh": (["random"], "rayleigh"),
    "sample": (["random"], "sample"),
    "standard_cauchy": (["random"], "standard_cauchy"),
    "standard_exponential": (["random"], "standard_exponential"),
    "standard_gamma": (["random"], "standard_gamma"),
    "standard_normal": (["random"], "standard_normal"),
    "uniform": (["random"], "uniform"),
    "weibull": (["random"], "weibull"),
    # linalg
    "cholesky": (["linalg"], "cholesky"),
    "det": (["linalg"], "det"),
    "dot": (["numpy"], "dot"),
    "eig": (["linalg"], "eig"),
    "eigvals": (["linalg"], "eigvals"),
    "matmul": (["numpy"], "matmul"),
    "matrix_power": (["linalg"], "matrix_power"),
    "matrix_rank": (["linalg"], "matrix_rank"),
    "multi_dot": (["linalg"], "multi_dot"),
    # transcendentals
    "nanprod": (["numpy"], "nanprod"),
    "nansum": (["numpy"], "nansum"),
    "prod": (["numpy"], "prod"),
    "sum": (["numpy"], "sum"),
    # array creations
    "full": (["numpy"], "full"),
    "ones_like": (["numpy"], "ones_like"),
    "zeros_like": (["numpy"], "zeros_like"),
    "full_like": (["numpy"], "full_like"),
    # array ops
    "copy": (["numpy"], "copy"),
    "cumsum": (["numpy"], "cumsum"),
    "cumprod": (["numpy"], "cumprod"),
    "sort": (["numpy"], "sort"),
    "take": (["numpy"], "take"),
}


class _RewriteNumPyOverloadedFunctionsImpl(object):
    def __init__(
        self, state, rewrite_function_name_map=rewrite_function_name_map
    ):
        self.state = state
        self.function_name_map = rewrite_function_name_map

    def run(self):
        """
        This function rewrites the name of NumPy functions that exist in self.function_name_map
        e.g np.sum(a) would produce the following:

        np.sum() --> numba_dpex.dpnp.sum()

        ---------------------------------------------------------------------------------------
        Numba IR Before Rewrite:
        ---------------------------------------------------------------------------------------

            $2load_global.0 = global(np: <module 'numpy' from 'numpy/__init__.py'>) ['$2load_global.0']
            $4load_method.1 = getattr(value=$2load_global.0, attr=sum) ['$2load_global.0', '$4load_method.1']
            $8call_method.3 = call $4load_method.1(a, func=$4load_method.1, args=[Var(a, test_rewrite.py:7)],
                                                   kws=(), vararg=None) ['$4load_method.1', '$8call_method.3', 'a']

        ---------------------------------------------------------------------------------------
        Numba IR After Rewrite:
        ---------------------------------------------------------------------------------------

            $dpex_replaced_var.0 = global(numba_dpex: <module 'numba_dpex' from 'numba_dpex/__init__.py'>) ['$dpex_replaced_var.0']
            $dpnp_var.1 = getattr(value=$dpex_replaced_var.0, attr=dpnp) ['$dpnp_var.1', '$dpex_replaced_var.0']
            $4load_method.1 = getattr(value=$dpnp_var.1, attr=sum) ['$4load_method.1', '$dpnp_var.1']
            $8call_method.3 = call $4load_method.1(a, func=$4load_method.1, args=[Var(a, test_rewrite.py:7)],
                                                   kws=(), vararg=None) ['$4load_method.1', '$8call_method.3', 'a']

        ---------------------------------------------------------------------------------------
        """
        func_ir = self.state.func_ir
        blocks = func_ir.blocks
        topo_order = find_topo_order(blocks)
        replaced = False
        for label in topo_order:
            block = blocks[label]
            saved_arr_arg = {}
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(
                    stmt.value, ir.Expr
                ):
                    lhs = stmt.target.name
                    # print("lhs.name= ",lhs)
                    rhs = stmt.value
                    # print("rhs.op= ", rhs.op)
                    # replace np.FOO with name from self.function_name_map["FOO"]
                    # e.g. np.sum will be replaced with numba_dpex.dpnp.sum
                    if (
                        rhs.op == "getattr"
                        and rhs.attr in self.function_name_map
                    ):
                        module_node = block.find_variable_assignment(
                            rhs.value.name
                        ).value
                        if (
                            isinstance(module_node, ir.Global)
                            and module_node.value.__name__
                            in self.function_name_map[rhs.attr][0]
                        ) or (
                            isinstance(module_node, ir.Expr)
                            and module_node.attr
                            in self.function_name_map[rhs.attr][0]
                        ):
                            rhs = stmt.value
                            rhs.attr = self.function_name_map[rhs.attr][1]

                            global_module = rhs.value
                            saved_arr_arg[lhs] = global_module

                            scope = global_module.scope
                            loc = global_module.loc

                            g_dpex_var = ir.Var(
                                scope, mk_unique_var("$2load_global"), loc
                            )
                            # We are trying to rename np.function_name/np.linalg.function_name with
                            # numba_dpex.dpnp.function_name.
                            # Hence, we need to have a global variable representing module numba_dpex.
                            # Next, we add attribute dpnp to global module numba_dpex to
                            # represent numba_dpex.dpnp.
                            g_dpex = ir.Global("numba_dpex", numba_dpex, loc)
                            g_dpex_assign = ir.Assign(g_dpex, g_dpex_var, loc)

                            dpnp_var = ir.Var(
                                scope, mk_unique_var("$4load_attr"), loc
                            )
                            getattr_dpnp = ir.Expr.getattr(
                                g_dpex_var, "dpnp", loc
                            )
                            dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                            rhs.value = dpnp_var
                            new_body.append(g_dpex_assign)
                            new_body.append(dpnp_assign)
                            func_ir._definitions[dpnp_var.name] = [getattr_dpnp]
                            func_ir._definitions[g_dpex_var.name] = [g_dpex]
                            replaced = True

                new_body.append(stmt)
            block.body = new_body
            return replaced


@register_pass(mutates_CFG=True, analysis_only=False)
class RewriteOverloadedNumPyFunctionsPass(FunctionPass):
    _name = "dpex_rewrite_overloaded_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

        import numba_dpex.dpnp_iface.dpnp_array_creations_impl
        import numba_dpex.dpnp_iface.dpnp_array_ops_impl
        import numba_dpex.dpnp_iface.dpnp_indexing
        import numba_dpex.dpnp_iface.dpnp_linalgimpl
        import numba_dpex.dpnp_iface.dpnp_logic
        import numba_dpex.dpnp_iface.dpnp_manipulation
        import numba_dpex.dpnp_iface.dpnp_randomimpl
        import numba_dpex.dpnp_iface.dpnp_sort_search_countimpl
        import numba_dpex.dpnp_iface.dpnp_statisticsimpl
        import numba_dpex.dpnp_iface.dpnp_transcendentalsimpl
        import numba_dpex.dpnp_iface.dpnpdecl
        import numba_dpex.dpnp_iface.dpnpimpl

    def run_pass(self, state):
        rewrite_function_name_pass = _RewriteNumPyOverloadedFunctionsImpl(
            state, rewrite_function_name_map
        )

        mutated = rewrite_function_name_pass.run()

        if mutated:
            remove_dead(
                state.func_ir.blocks, state.func_ir.arg_names, state.func_ir
            )
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return mutated


def get_dpnp_func_typ(func):
    from numba.core.typing.templates import builtin_registry

    for (k, v) in builtin_registry.globals:
        if k == func:
            return v
    raise RuntimeError("type for func ", func, " not found")


class _RewriteNdarrayFunctionsImpl(object):
    def __init__(
        self, state, rewrite_function_name_map=rewrite_function_name_map
    ):
        self.state = state
        self.function_name_map = rewrite_function_name_map
        self.typemap = state.type_annotation.typemap
        self.calltypes = state.type_annotation.calltypes

    def run(self):
        typingctx = self.state.typingctx

        # save array arg to call
        # call_varname -> array
        func_ir = self.state.func_ir
        blocks = func_ir.blocks
        saved_arr_arg = {}
        topo_order = find_topo_order(blocks)
        replaced = False

        for label in topo_order:
            block = blocks[label]
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(
                    stmt.value, ir.Expr
                ):
                    lhs = stmt.target.name
                    rhs = stmt.value
                    # replace A.func with np.func, and save A in saved_arr_arg
                    if (
                        rhs.op == "getattr"
                        and rhs.attr in self.function_name_map
                        and isinstance(
                            self.typemap[rhs.value.name], types.npytypes.Array
                        )
                    ):
                        rhs = stmt.value
                        arr = rhs.value
                        saved_arr_arg[lhs] = arr
                        scope = arr.scope
                        loc = arr.loc

                        g_dpex_var = ir.Var(
                            scope, mk_unique_var("$load_global"), loc
                        )
                        self.typemap[g_dpex_var.name] = types.misc.Module(
                            numba_dpex
                        )
                        g_dpex = ir.Global("numba_dpex", numba_dpex, loc)
                        g_dpex_assign = ir.Assign(g_dpex, g_dpex_var, loc)

                        dpnp_var = ir.Var(
                            scope, mk_unique_var("$load_attr"), loc
                        )
                        self.typemap[dpnp_var.name] = types.misc.Module(
                            numba_dpex.dpnp
                        )
                        getattr_dpnp = ir.Expr.getattr(g_dpex_var, "dpnp", loc)
                        dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                        rhs.value = dpnp_var
                        new_body.append(g_dpex_assign)
                        new_body.append(dpnp_assign)

                        func_ir._definitions[g_dpex_var.name] = [getattr_dpnp]
                        func_ir._definitions[dpnp_var.name] = [getattr_dpnp]

                        # update func var type
                        func = getattr(numba_dpex.dpnp, rhs.attr)
                        func_typ = get_dpnp_func_typ(func)

                        self.typemap.pop(lhs)
                        self.typemap[lhs] = func_typ
                        replaced = True

                    if rhs.op == "call" and rhs.func.name in saved_arr_arg:
                        # add array as first arg
                        arr = saved_arr_arg[rhs.func.name]
                        # update call type signature to include array arg
                        old_sig = self.calltypes.pop(rhs)
                        # argsort requires kws for typing so sig.args can't be used
                        # reusing sig.args since some types become Const in sig
                        argtyps = old_sig.args[: len(rhs.args)]
                        kwtyps = {
                            name: self.typemap[v.name] for name, v in rhs.kws
                        }
                        self.calltypes[rhs] = self.typemap[
                            rhs.func.name
                        ].get_call_type(
                            typingctx,
                            [self.typemap[arr.name]] + list(argtyps),
                            kwtyps,
                        )
                        rhs.args = [arr] + rhs.args

                new_body.append(stmt)
            block.body = new_body
        return replaced


@register_pass(mutates_CFG=True, analysis_only=False)
class RewriteNdarrayFunctionsPass(FunctionPass):
    _name = "dpex_rewrite_ndarray_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        rewrite_ndarray_function_name_pass = _RewriteNdarrayFunctionsImpl(
            state, rewrite_function_name_map
        )

        mutated = rewrite_ndarray_function_name_pass.run()

        if mutated:
            remove_dead(
                state.func_ir.blocks, state.func_ir.arg_names, state.func_ir
            )
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return mutated


class _IdentifyNumPyFunctionsPassImpl(object):
    def __init__(
        self, state, rewrite_function_name_map=rewrite_function_name_map
    ):
        self.state = state
        self.function_name_map = rewrite_function_name_map

    def run(self):

        func_ir = self.state.func_ir
        blocks = func_ir.blocks
        topo_order = find_topo_order(blocks)

        for label in topo_order:
            block = blocks[label]
            numOp = 0
            numNp = 0
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(
                    stmt.value, ir.Expr
                ):

                    rhs = stmt.value

                    if rhs.op == "call":
                        numOp += 1
                        name = rhs.func.name
                        rhs = get_definition(func_ir, name)

                        while not isinstance(rhs, ir.Global):
                            rhs = get_definition(func_ir, rhs.value)

                        print(
                            "Call ops ",
                            numOp,
                            " : ",
                            name,
                            " is loaded from ",
                            rhs.value,
                        )
                        if "numpy" in str(rhs):
                            numNp += 1

            print("Num of Instructions (IR nodes) = ", len(block.body))
            print("Num of Call Operations         = ", numOp)
            print("Num of Numpy Functions         = ", numNp)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class IdentifyNumPyFunctionsPass(FunctionPass):
    _name = "dpex_count_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        rewrite_function_name_pass = _IdentifyNumPyFunctionsPassImpl(
            state, rewrite_function_name_map
        )

        mutated = rewrite_function_name_pass.run()

        if mutated:
            remove_dead(
                state.func_ir.blocks, state.func_ir.arg_names, state.func_ir
            )
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return mutated
