# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import PassManager
from numba.core.typed_passes import (
    AnnotateTypes,
    InlineOverloads,
    IRLegalization,
    NativeParforLowering,
    NopythonRewrites,
    NoPythonSupportedFeatureValidation,
    NopythonTypeInference,
    ParforFusionPass,
    ParforPreLoweringPass,
    PreLowerStripPhis,
    PreParforPass,
)

from numba_dpex.core.exceptions import UnsupportedCompilationModeError
from numba_dpex.core.parfors.parfor_diagnostics import ExtendedParforDiagnostics
from numba_dpex.core.parfors.parfor_pass import ParforPass
from numba_dpex.core.passes import (
    DumpParforDiagnostics,
    NoPythonBackend,
    ParforLegalizeCFDPass,
)


class _DpjitPassBuilder(object):
    """
    A pass builder for dpex's DpjitCompiler that adds supports for
    offloading dpnp array expressions and dpnp library calls to a SYCL device.

    The pass builder does not implement pipelines for objectmode or interpreted
    execution.
    """

    @staticmethod
    def define_typed_pipeline(state, name="dpex_dpjit_typed"):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        # Annotate only once legalized
        pm.add_pass(AnnotateTypes, "annotate types")
        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimization
        pm.add_pass(InlineOverloads, "inline overloaded functions")
        pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        pm.add_pass(ParforPass, "convert to parfors")
        pm.add_pass(
            ParforLegalizeCFDPass,
            "Legalize parfors for compute follows data",
        )
        pm.add_pass(ParforFusionPass, "fuse parfors")
        pm.add_pass(ParforPreLoweringPass, "parfor prelowering")

        pm.finalize()
        return pm

    @classmethod
    def define_nopython_lowering_pipeline(
        cls, state, name="dpex_dpjit_lowering"
    ):
        """Returns an nopython mode pipeline based PassManager"""
        pm = PassManager(name)

        # legalize
        pm.add_pass(
            NoPythonSupportedFeatureValidation,
            "ensure features that are in use are in a valid form",
        )
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(
            NativeParforLowering, "lowerer with support for parfor nodes"
        )
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")

        pm.finalize()
        return pm

    @classmethod
    def define_nopython_pipeline(cls, state, name="dpex_dpjit_nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        # compose pipeline from untyped, typed and lowering parts
        dpb = cls
        pm = PassManager(name)
        untyped_passes = DefaultPassBuilder.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm


class DpjitCompiler(CompilerBase):
    """Dpex's compiler pipeline to offload parfor nodes into SYCL kernels."""

    _pass_builder = _DpjitPassBuilder

    def define_pipelines(self):
        pms = []
        self.state.parfor_diagnostics = ExtendedParforDiagnostics()
        self.state.metadata["parfor_diagnostics"] = (
            self.state.parfor_diagnostics
        )
        if not self.state.flags.force_pyobject:
            pms.append(self._pass_builder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            raise UnsupportedCompilationModeError()
        return pms
