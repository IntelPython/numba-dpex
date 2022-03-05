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

from llvmlite import binding as ll
from llvmlite.llvmpy import core as lc
from numba.core import utils
from numba.core.codegen import CPUCodegen, CPUCodeLibrary

from numba_dpex import config

SPIR_TRIPLE = {32: " spir-unknown-unknown", 64: "spir64-unknown-unknown"}

SPIR_DATA_LAYOUT = {
    32: (
        "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
        "256-v512:512-v1024:1024"
    ),
    64: (
        "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
        "v512:512-v1024:1024"
    ),
}


class SPIRVCodeLibrary(CPUCodeLibrary):
    def _optimize_functions(self, ll_module):
        pass

    def _optimize_final_module(self):
        # Run some lightweight optimization to simplify the module.
        pmb = ll.PassManagerBuilder()

        # Make optimization level depending on config.OPT variable
        pmb.opt_level = config.OPT

        pmb.disable_unit_at_a_time = False
        pmb.disable_unroll_loops = True
        pmb.loop_vectorize = False
        pmb.slp_vectorize = False

        pm = ll.ModulePassManager()
        pmb.populate(pm)
        pm.run(self._final_module)

    def _finalize_specific(self):
        # Fix global naming
        for gv in self._final_module.global_variables:
            if "." in gv.name:
                gv.name = gv.name.replace(".", "_")

    def get_asm_str(self):
        # Return nothing: we can only dump assembler code when it is later
        # generated (in numba_dpex.compiler).
        return None


class JITSPIRVCodegen(CPUCodegen):
    """
    This codegen implementation generates optimized SPIR 2.0
    """

    _library_class = SPIRVCodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = SPIR_DATA_LAYOUT[utils.MACHINE_BITS]
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = lc.Module(name)
        ir_module.triple = SPIR_TRIPLE[utils.MACHINE_BITS]
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    def _module_pass_manager(self):
        raise NotImplementedError

    def _function_pass_manager(self, llvm_module):
        raise NotImplementedError

    def _add_module(self, module):
        pass
