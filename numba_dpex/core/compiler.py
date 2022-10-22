# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.compiler import CompilerBase, DefaultPassBuilder

from numba_dpex.parfor_diagnostics import ExtendedParforDiagnostics

from .passbuilder import PassBuilder


class Compiler(CompilerBase):
    """The DPEX compiler pipeline."""

    def define_pipelines(self):
        # this maintains the objmode fallback behaviour
        pms = []
        self.state.parfor_diagnostics = ExtendedParforDiagnostics()
        self.state.metadata[
            "parfor_diagnostics"
        ] = self.state.parfor_diagnostics
        if not self.state.flags.force_pyobject:
            pms.append(PassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            pms.append(
                DefaultPassBuilder.define_objectmode_pipeline(self.state)
            )
        return pms
