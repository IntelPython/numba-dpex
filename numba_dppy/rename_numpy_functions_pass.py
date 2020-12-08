from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (
    find_topo_order,
    mk_unique_var,
    remove_dead,
    simplify_CFG,
)
import numba_dppy

rewrite_function_name_map = {"sum": (["np"], "sum"), "eig": (["linalg"], "eig")}


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
                            and module_node.name in self.function_name_map[rhs.attr][0]
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

                new_body.append(stmt)
            block.body = new_body


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYRewriteOverloadedFunctions(FunctionPass):
    _name = "dppy_rewrite_overloaded_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)
        import numba_dppy.dpnp_glue.dpnpdecl
        import numba_dppy.dpnp_glue.dpnpimpl

    def run_pass(self, state):
        rewrite_function_name_pass = RewriteNumPyOverloadedFunctions(
            state, rewrite_function_name_map
        )

        rewrite_function_name_pass.run()

        remove_dead(state.func_ir.blocks, state.func_ir.arg_names, state.func_ir)
        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return True
