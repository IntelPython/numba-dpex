#! /usr/bin/env python

import unittest
import numpy as np

import numba
from numba import njit, prange
import numba_dppy, numba_dppy as dppy


from numba.core import compiler
from numba_dppy.rename_numpy_functions_pass import DPPYRewriteOverloadedFunctions


class MyPipeline(object):
    def __init__(self, test_ir):
        self.state = compiler.StateDict()
        self.state.func_ir = test_ir


class TestRenameNumpyFunctionsPass(unittest.TestCase):
    def test_rename(self):
        def expected(a):
            b = numba_dppy.dpnp.sum(a)
            return b

        def got(a):
            b = np.sum(a)
            return b

        expected_ir = compiler.run_frontend(expected)
        got_ir = compiler.run_frontend(got)

        pipeline = MyPipeline(got_ir)

        rewrite_numpy_functions_pass = DPPYRewriteOverloadedFunctions()
        rewrite_numpy_functions_pass.run_pass(pipeline.state)

        self.assertEqual(got_ir, pipeline.state.func_ir)


if __name__ == "__main__":
    unittest.main()
