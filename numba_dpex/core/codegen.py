# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

from llvmlite import binding as ll
from llvmlite import ir as llvmir
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

    @property
    def inline_threshold(self):
        """
        The inlining threshold value to be used to optimize the final library.
        """
        if hasattr(self, "_inline_threshold"):
            return self._inline_threshold
        else:
            return 0

    @inline_threshold.setter
    def inline_threshold(self, value: int):
        """Returns the current inlining threshold level for the library."""
        if value < 0 or value > 3:
            warnings.warn(
                "Unsupported inline threshold. Set a value between 0 and 3"
            )
            self._inline_threshold = 0
        else:
            if value == 3:
                warnings.warn(
                    "Due to an existing compiler bug, setting INLINE_THRESHOLD "
                    f"to {value} can lead to incorrect code generation on "
                    "certain devices."
                )
            self._inline_threshold = value

    def _optimize_final_module(self):
        # Run some lightweight optimization to simplify the module.
        pmb = ll.PassManagerBuilder()

        # Make optimization level depending on config.DPEX_OPT variable
        pmb.opt_level = config.DPEX_OPT
        if config.DPEX_OPT > 2:
            warnings.warn(
                "Setting NUMBA_DPEX_OPT greater than 2 known to cause issues "
                + "related to very aggressive optimizations that leads to "
                + "broken code."
            )

        pmb.disable_unit_at_a_time = False
        pmb.inlining_threshold = self.inline_threshold

        pmb.disable_unroll_loops = True
        pmb.loop_vectorize = False
        pmb.slp_vectorize = False

        pm = ll.ModulePassManager()
        pmb.populate(pm)
        pm.run(self._final_module)

    def optimize_final_module(self):
        """Public member function to optimize the final LLVM module in the
        library. The function calls the protected overridden function.
        """
        self._optimize_final_module()

    def _finalize_specific(self):
        # Fix global naming
        for gv in self._final_module.global_variables:
            if "." in gv.name:
                gv.name = gv.name.replace(".", "_")

    def get_asm_str(self):
        # Return nothing: we can only dump assembler code when it is later
        # generated (in numba_dpex.compiler).
        return None

    @property
    def final_module(self):
        return self._final_module


class JITSPIRVCodegen(CPUCodegen):
    """
    This codegen implementation generates optimized SPIR 2.0
    """

    _library_class = SPIRVCodeLibrary

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"
        self._data_layout = SPIR_DATA_LAYOUT[utils.MACHINE_BITS]
        self._target_data = ll.create_target_data(self._data_layout)
        self._tm_features = (
            ""  # We need this for caching, not sure about this value for now
        )

    def _create_empty_module(self, name):
        ir_module = llvmir.Module(name)
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
