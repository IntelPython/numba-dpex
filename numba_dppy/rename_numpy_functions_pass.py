from numba.core import ir
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import (find_topo_order, mk_unique_var,
                                 simplify_CFG)

rewrite_function_name_map = {"sum": (["np"], "sum"),
                             "eig": (["linalg"], "eig")}

class RewriteOverloadedFunctions(object):
    def __init__(self, state, rewrite_function_name_map=rewrite_function_name_map):
        self.state = state
        self.function_name_map = rewrite_function_name_map

    def run(self):
        func_ir = self.state.func_ir
        blocks = func_ir.blocks
        topo_order = find_topo_order(blocks)

        import numba_dppy.dpnp_glue.dpnpdecl
        import numba_dppy.dpnp_glue.dpnpimpl
        for label in topo_order:
            block = blocks[label]
            saved_arr_arg = {}
            new_body = []
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
                    lhs = stmt.target.name
                    rhs = stmt.value
                    # replace np.func with name from map np.func
                    if (rhs.op == 'getattr' and rhs.attr in self.function_name_map):
                        module_node = block.find_variable_assignment(rhs.value.name).value
                        if ((isinstance(module_node, ir.Global) and
                                module_node.name in self.function_name_map[rhs.attr][0]) or
                            (isinstance(module_node, ir.Expr) and
                                module_node.attr in self.function_name_map[rhs.attr][0])):
                            rhs = stmt.value
                            rhs.attr = self.function_name_map[rhs.attr][1]

                            global_module = rhs.value
                            saved_arr_arg[lhs] = global_module

                            scope = global_module.scope
                            loc = global_module.loc

                            g_dppy_var = ir.Var(scope, mk_unique_var("$dppy_replaced_var"), loc)
                            g_dppy = ir.Global('numba_dppy', numba_dppy, loc)
                            g_dppy_assign = ir.Assign(g_dppy, g_dppy_var, loc)

                            dpnp_var = ir.Var(scope, mk_unique_var("$dpnp_var"), loc)
                            getattr_dpnp = ir.Expr.getattr(g_dppy_var, 'dpnp', loc)
                            dpnp_assign = ir.Assign(getattr_dpnp, dpnp_var, loc)

                            rhs.value = dpnp_var
                            new_body.append(g_dppy_assign)
                            new_body.append(dpnp_assign)
                            func_ir._definitions[dpnp_var.name] = [getattr_dpnp]
                            func_ir._definitions[g_dppy_var.name] = [g_dppy]

                new_body.append(stmt)
            block.body = new_body
            import pdb
            pdb.set_trace()


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPYRewriteOverloadedFunctions(FunctionPass):
    _name = "dppy_rewrite_overloaded_functions_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        rewrite_function_name_pass = RewriteOverloadedFunctions(state, rewrite_function_name_map)

        rewrite_function_name_pass.run()

        state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        return True
