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

from numba.core.debuginfo import DIBuilder
from llvmlite import ir


class DPPYDIBuilder(DIBuilder):
    def __init__(self, module, filepath, linkage_name):
        DIBuilder.__init__(self, module, filepath)
        self.linkage_name = linkage_name

    def mark_subprogram(self, function, name, loc):
        di_subp = self._add_subprogram(
            name=name, linkagename=self.linkage_name, line=loc.line
        )
        function.set_metadata("dbg", di_subp)
        # disable inlining for this function for easier debugging
        function.attributes.add("noinline")

    def _di_compile_unit(self):
        di = super()._di_compile_unit()
        operands = dict(di.operands)
        operands["language"] = ir.DIToken("DW_LANG_C_plus_plus")
        operands["producer"] = "numba-dppy"
        di.operands = tuple(operands.items())
        return di
