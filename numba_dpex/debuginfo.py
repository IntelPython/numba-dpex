# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import numba as nb
from llvmlite import ir
from numba.core.debuginfo import DIBuilder

from numba_dpex.numba_support import numba_version


class DpexDIBuilder(DIBuilder):
    def __init__(self, module, filepath, linkage_name, cgctx):
        args = []

        if numba_version > (0, 54):
            args.append(cgctx)

        DIBuilder.__init__(self, module, filepath, *args)
        self.linkage_name = linkage_name

    if numba_version > (0, 54):

        def mark_subprogram(self, function, qualname, argnames, argtypes, line):
            name = qualname
            argmap = dict(zip(argnames, argtypes))
            di_subp = self._add_subprogram(
                name=name,
                linkagename=self.linkage_name,
                line=line,
                function=function,
                argmap=argmap,
            )
            function.set_metadata("dbg", di_subp)
            # disable inlining for this function for easier debugging
            function.attributes.add("noinline")

    else:

        def mark_subprogram(self, function, name, line):
            di_subp = self._add_subprogram(
                name=name, linkagename=self.linkage_name, line=line
            )
            function.set_metadata("dbg", di_subp)
            # disable inlining for this function for easier debugging
            function.attributes.add("noinline")

    def _di_compile_unit(self):
        di = super()._di_compile_unit()
        operands = dict(di.operands)
        operands["language"] = ir.DIToken("DW_LANG_C_plus_plus")
        operands["producer"] = "numba-dpex"
        di.operands = tuple(operands.items())
        return di
