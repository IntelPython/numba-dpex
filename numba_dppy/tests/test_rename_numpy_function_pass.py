#! /usr/bin/env python
import unittest
import numpy as np
import numba
from numba.core import compiler
from numba_dppy.rename_numpy_functions_pass import DPPYRewriteOverloadedFunctions


class MyPipeline(object):
    def __init__(self, test_ir):
        self.state = compiler.StateDict()
        self.state.func_ir = test_ir


def check_equivalent(expected_ir, got_ir):
    expected_block_body = expected_ir.blocks[0].body
    got_block_body = got_ir.blocks[0].body

    if len(expected_block_body) != len(got_block_body):
        return False

    for i in range(len(expected_block_body)):
        expected_stmt = expected_block_body[i]
        got_stmt = got_block_body[i]
        if type(expected_stmt) != type(got_stmt):
            return False
        else:
            if isinstance(expected_stmt, numba.core.ir.Assign):
                if isinstance(expected_stmt.value, numba.core.ir.Global):
                    if (expected_stmt.value.name != got_stmt.value.name and
                        expected_stmt.value.name != "numba_dppy"):
                        return False
                elif isinstance(expected_stmt.value, numba.core.ir.Expr):
                    # should get "dpnp" and "sum" as attr
                    if expected_stmt.value.op == "getattr":
                        if expected_stmt.value.attr != got_stmt.value.attr:
                            return False
    return True


class TestRenameNumpyFunctionsPass(unittest.TestCase):
    def test_rename(self):
        def expected(a):
            return numba_dppy.dpnp.sum(a)

        def got(a):
            return np.sum(a)

        expected_ir = compiler.run_frontend(expected)
        got_ir = compiler.run_frontend(got)

        pipeline = MyPipeline(got_ir)

        rewrite_numpy_functions_pass = DPPYRewriteOverloadedFunctions()
        rewrite_numpy_functions_pass.run_pass(pipeline.state)

        self.assertTrue(check_equivalent(expected_ir, pipeline.state.func_ir))


if __name__ == "__main__":
    unittest.main()
