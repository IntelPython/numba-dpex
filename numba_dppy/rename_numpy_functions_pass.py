from numba.core import ir, compiler
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (
    find_topo_order,
    mk_unique_var,
    remove_dead,
    simplify_CFG,
    get_name_var_table,
)
import numba_dppy
from numba.core import types
import numpy as np

rewrite_function_name_map = {
    "sum": (["np"], "sum"),
    "eig": (["linalg"], "eig"),
    "prod": (["np"], "prod"),
    "max": (["np"], "max"),
    "amax": (["np"], "amax"),
    "min": (["np"], "min"),
    "amin": (["np"], "amin"),
    "mean": (["np"], "mean"),
    "median": (["np"], "median"),
    "argmax": (["np"], "argmax"),
    "argmin": (["np"], "argmin"),
    "argsort": (["np"], "argsort"),
    "cov": (["np"], "cov"),
    "dot": (["np"], "dot"),
    "matmul": (["np"], "matmul"),
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
        saved_arr_arg = {}

        for label in topo_order:
            block = blocks[label]
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
                            and module_node.name in self.function_name_map[rhs.attr][0]
                        ) or (
                            isinstance(module_node, ir.Expr)
                            and module_node.attr in self.function_name_map[rhs.attr][0]
                        ):
                            rhs = stmt.value
                            saved_arr_arg[lhs] = rhs.value
                            rhs.attr = self.function_name_map[rhs.attr][1]

                            global_module = rhs.value

                            scope = global_module.scope
                            loc = global_module.loc

                            g_dppy_var = ir.Var(
                                scope, mk_unique_var("$load_global"), loc
                            )
                            # We are trying to rename np.function_name/np.linalg.function_name with
                            # numba_dppy.dpnp.function_name.
                            # Hence, we need to have a global variable representing module numba_dppy.
                            # Next, we add attribute dpnp to global module numba_dppy to
                            # represent numba_dppy.dpnp.
                            g_dppy = ir.Global("numba_dppy", numba_dppy, loc)
                            g_dppy_assign = ir.Assign(g_dppy, g_dppy_var, loc)

                            dpnp_var = ir.Var(scope, mk_unique_var("$load_attr"), loc)
                            getattr_dpnp = ir.Expr.getattr(g_dppy_var, "dpnp", loc)
                            dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                            rhs.value = dpnp_var
                            new_body.append(g_dppy_assign)
                            new_body.append(dpnp_assign)
                            func_ir._definitions[dpnp_var.name] = [getattr_dpnp]
                            func_ir._definitions[g_dppy_var.name] = [g_dppy]

                new_body.append(stmt)
            block.body = new_body


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

    def run_pass(self, state):
        rewrite_function_name_pass = RewriteNumPyOverloadedFunctions(
            state, rewrite_function_name_map
        )

        rewrite_function_name_pass.run()

        remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return True


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
        return


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYRewriteNdarrayFunctions(FunctionPass):
    _name = "dppy_rewrite_ndarray_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        rewrite_ndarray_function_name_pass = RewriteNdarrayFunctions(
            state, rewrite_function_name_map
        )

        rewrite_ndarray_function_name_pass.run()

        remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return True


class ConvertDPNPArgs(object):
    def __init__(self, state, rewrite_function_name_map=rewrite_function_name_map):
        self.state = state
        self.typemap = state.type_annotation.typemap
        self.calltypes = state.type_annotation.calltypes
        self.function_name_map = rewrite_function_name_map

    def run(self):
        """
        In this pass we find out call nodes that belong to dpnp submodule.
        If we find any, for each argument of that call node that is of types.npytypes.Array,
        we add a call to a convert function. The convert function will ideally
        check if the ndarray is USM backed and if not will create a new ndarray that is
        USM backed and copy the exisiting data into the new array.

        """
        typingctx = self.state.typingctx
        func_id = 0

        # save array arg to call
        # call_varname -> array
        func_ir = self.state.func_ir
        blocks = func_ir.blocks
        saved_arr_arg = {}
        topo_order = find_topo_order(blocks)

        for label in topo_order:
            block = blocks[label]
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    lhs = stmt.target.name
                    rhs = stmt.value

                    if rhs.op == "getattr" and rhs.attr in self.function_name_map:
                        module_node = block.find_variable_assignment(
                            rhs.value.name
                        ).value
                        if (
                            isinstance(module_node, ir.Expr) and module_node.op == "getattr"
                            and module_node.attr == "dpnp"
                            ):
                            scope = rhs.value.scope
                            loc = rhs.value.loc

                            saved_arr_arg[lhs] = rhs.value
                            convert_var = ir.Var(scope, mk_unique_var("$load_method"), loc)
                            saved_arr_arg["convert_var"] = convert_var
                            getattr_convert = ir.Expr.getattr(rhs.value, "convert_ndarray_to_usm",
                                                              loc)
                            convert_assign = ir.Assign(getattr_convert, convert_var, loc)
                            self.typemap[convert_var.name] = get_dpnp_func_typ(getattr(numba_dppy.dpnp,
                                                                              getattr_convert.attr))
                            func_ir._definitions[convert_var.name] = [getattr_convert]

                            new_body.append(convert_assign)

                    if rhs.op == "call" and rhs.func.name in saved_arr_arg:
                        # the assumption hwew is that we only get the ndarray as the argument
                        new_args_list = list()
                        for arg in rhs.args:

                            if isinstance(self.typemap[arg.name], types.npytypes.Array):
                                convert_func_var = ir.Var(scope, mk_unique_var("$call_method"), loc)
                                new_args_list.append(convert_func_var)
                                convert_func_expr = ir.Expr.call(func=saved_arr_arg["convert_var"],
                                                                 args=[arg], kws=(), loc=loc)
                                convert_func_assign = ir.Assign(convert_func_expr, convert_func_var, loc)
                                self.typemap[convert_func_var.name] = self.typemap[arg.name]
                                new_body.append(convert_func_assign)

                                self.calltypes[convert_func_expr] = self.typemap[saved_arr_arg["convert_var"].name].get_call_type(typingctx, [self.typemap[arg.name]], {})
                            else:
                                new_args_list.append(arg)

                        rhs.args = new_args_list

                new_body.append(stmt)
            block.body = new_body

        return


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYConvertDPNPArgumentsToUSM(FunctionPass):
    _name = "dppy_convert_dpnp_function_arguments_to_usm_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        convert_dpnp_args_pass = ConvertDPNPArgs(state)

        convert_dpnp_args_pass.run()

        remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return True
