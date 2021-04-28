# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (
    find_topo_order,
    mk_unique_var,
    remove_dead,
    simplify_CFG,
)
import numba_dppy
from numba.core import types


rewrite_function_name_map = {
    # numpy
    "amax": (["numpy"], "amax"),
    "amin": (["numpy"], "amin"),
    "argmax": (["numpy"], "argmax"),
    "argmin": (["numpy"], "argmin"),
    "argsort": (["numpy"], "argsort"),
    "cov": (["numpy"], "cov"),
    "max": (["numpy"], "max"),
    "mean": (["numpy"], "mean"),
    "median": (["numpy"], "median"),
    "min": (["numpy"], "min"),
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


class RewriteNumPyOverloadedFunctions(object):
    def __init__(self, state, rewrite_function_name_map=rewrite_function_name_map):
        self.state = state
        self.function_name_map = rewrite_function_name_map

    def run(self):
        """
        This function rewrites the name of NumPy functions that exist in self.function_name_map
        e.g np.sum(a) would produce the following:

        np.sum() --> numba_dppy.dpnp.sum()

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

            $dppy_replaced_var.0 = global(numba_dppy: <module 'numba_dppy' from 'numba_dppy/__init__.py'>) ['$dppy_replaced_var.0']
            $dpnp_var.1 = getattr(value=$dppy_replaced_var.0, attr=dpnp) ['$dpnp_var.1', '$dppy_replaced_var.0']
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
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    lhs = stmt.target.name
                    rhs = stmt.value
                    # replace np.FOO with name from self.function_name_map["FOO"]
                    # e.g. np.sum will be replaced with numba_dppy.dpnp.sum
                    if rhs.op == "getattr" and rhs.attr in self.function_name_map:
                        module_node = block.find_variable_assignment(
                            rhs.value.name
                        ).value
                        if (
                            isinstance(module_node, ir.Global)
                            and module_node.value.__name__
                            in self.function_name_map[rhs.attr][0]
                        ) or (
                            isinstance(module_node, ir.Expr)
                            and module_node.attr in self.function_name_map[rhs.attr][0]
                        ):
                            rhs = stmt.value
                            rhs.attr = self.function_name_map[rhs.attr][1]

                            global_module = rhs.value
                            saved_arr_arg[lhs] = global_module

                            scope = global_module.scope
                            loc = global_module.loc

                            g_dppy_var = ir.Var(
                                scope, mk_unique_var("$2load_global"), loc
                            )
                            # We are trying to rename np.function_name/np.linalg.function_name with
                            # numba_dppy.dpnp.function_name.
                            # Hence, we need to have a global variable representing module numba_dppy.
                            # Next, we add attribute dpnp to global module numba_dppy to
                            # represent numba_dppy.dpnp.
                            g_dppy = ir.Global("numba_dppy", numba_dppy, loc)
                            g_dppy_assign = ir.Assign(g_dppy, g_dppy_var, loc)

                            dpnp_var = ir.Var(scope, mk_unique_var("$4load_attr"), loc)
                            getattr_dpnp = ir.Expr.getattr(g_dppy_var, "dpnp", loc)
                            dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                            rhs.value = dpnp_var
                            new_body.append(g_dppy_assign)
                            new_body.append(dpnp_assign)
                            func_ir._definitions[dpnp_var.name] = [getattr_dpnp]
                            func_ir._definitions[g_dppy_var.name] = [g_dppy]
                            replaced = True

                new_body.append(stmt)
            block.body = new_body
            return replaced


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYRewriteOverloadedNumPyFunctions(FunctionPass):
    _name = "dppy_rewrite_overloaded_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

        import numba_dppy.dpnp_glue.dpnpdecl
        import numba_dppy.dpnp_glue.dpnpimpl
        import numba_dppy.dpnp_glue.dpnp_linalgimpl
        import numba_dppy.dpnp_glue.dpnp_transcendentalsimpl
        import numba_dppy.dpnp_glue.dpnp_statisticsimpl
        import numba_dppy.dpnp_glue.dpnp_sort_search_countimpl
        import numba_dppy.dpnp_glue.dpnp_randomimpl
        import numba_dppy.dpnp_glue.dpnp_array_creations_impl
        import numba_dppy.dpnp_glue.dpnp_array_ops_impl

    def run_pass(self, state):
        rewrite_function_name_pass = RewriteNumPyOverloadedFunctions(
            state, rewrite_function_name_map
        )

        mutated = rewrite_function_name_pass.run()

        if mutated:
            remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return mutated


def get_dpnp_func_typ(func):
    from numba.core.typing.templates import builtin_registry

    for (k, v) in builtin_registry.globals:
        if k == func:
            return v
    raise RuntimeError("type for func ", func, " not found")


class RewriteNdarrayFunctions(object):
    def __init__(self, state, rewrite_function_name_map=rewrite_function_name_map):
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
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
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

                        g_dppy_var = ir.Var(scope, mk_unique_var("$load_global"), loc)
                        self.typemap[g_dppy_var.name] = types.misc.Module(numba_dppy)
                        g_dppy = ir.Global("numba_dppy", numba_dppy, loc)
                        g_dppy_assign = ir.Assign(g_dppy, g_dppy_var, loc)

                        dpnp_var = ir.Var(scope, mk_unique_var("$load_attr"), loc)
                        self.typemap[dpnp_var.name] = types.misc.Module(numba_dppy.dpnp)
                        getattr_dpnp = ir.Expr.getattr(g_dppy_var, "dpnp", loc)
                        dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                        rhs.value = dpnp_var
                        new_body.append(g_dppy_assign)
                        new_body.append(dpnp_assign)

                        func_ir._definitions[g_dppy_var.name] = [getattr_dpnp]
                        func_ir._definitions[dpnp_var.name] = [getattr_dpnp]

                        # update func var type
                        func = getattr(numba_dppy.dpnp, rhs.attr)
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
                        kwtyps = {name: self.typemap[v.name] for name, v in rhs.kws}
                        self.calltypes[rhs] = self.typemap[rhs.func.name].get_call_type(
                            typingctx, [self.typemap[arr.name]] + list(argtyps), kwtyps
                        )
                        rhs.args = [arr] + rhs.args

                new_body.append(stmt)
            block.body = new_body
        return replaced


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYRewriteNdarrayFunctions(FunctionPass):
    _name = "dppy_rewrite_ndarray_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        rewrite_ndarray_function_name_pass = RewriteNdarrayFunctions(
            state, rewrite_function_name_map
        )

        mutated = rewrite_ndarray_function_name_pass.run()

        if mutated:
            remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return mutated
