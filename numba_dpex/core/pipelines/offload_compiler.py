# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.compiler import CompilerBase
from numba.core.compiler_machinery import PassManager
from numba.core.typed_passes import (
    AnnotateTypes,
    InlineOverloads,
    IRLegalization,
    NopythonRewrites,
    NoPythonSupportedFeatureValidation,
    NopythonTypeInference,
    PreLowerStripPhis,
)
from numba.core.untyped_passes import (
    DeadBranchPrune,
    FindLiterallyCalls,
    FixupArgs,
    GenericRewrites,
    InlineClosureLikes,
    InlineInlinables,
    IRProcessing,
    LiteralPropagationSubPipelinePass,
    LiteralUnroll,
    MakeFunctionToJitFunction,
    ReconstructSSA,
    RewriteSemanticConstants,
    TranslateByteCode,
    WithLifting,
)

from numba_dpex.core.exceptions import UnsupportedCompilationModeError
from numba_dpex.core.passes.passes import (
    DpexLowering,
    DumpParforDiagnostics,
    NoPythonBackend,
    ParforPass,
    PreParforPass,
)
from numba_dpex.core.passes.rename_numpy_functions_pass import (
    RewriteNdarrayFunctionsPass,
    RewriteOverloadedNumPyFunctionsPass,
)
from numba_dpex.parfor_diagnostics import ExtendedParforDiagnostics


class _OffloadPassBuilder(object):
    """
    A pass builder for dpex's OffloAdCompiler that adds supports for
    offloading parfors and other NumPy library calls to a SYCL device.

    The pass builder does not implement pipelines for objectmode or interpreted
    execution.
    """

    @staticmethod
    def define_untyped_pipeline(state, name="dpex_offload_untyped"):
        """Returns an untyped part of the nopython pipeline

        The dpex offload pipeline's untyped passes are based on Numba's
        compiler. Dpex addes extra passes to change specific numpy overloads
        for which dpex provides alternate implementations.
        """
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # --- Begin dpex passes added to the untyped pipeline                --#

        # The RewriteOverloadedNumPyFunctionsPass rewrites the module namespace
        # of specific NumPy functions to dpnp, as we overload these functions
        # differently.
        pm.add_pass(
            RewriteOverloadedNumPyFunctionsPass,
            "Rewrite name of Numpy functions to overload already overloaded "
            + "function",
        )
        # --- End of dpex passes added to the untyped pipeline               --#

        # inline closures early in case they are using nonlocal's
        # see issue #6585.
        pm.add_pass(
            InlineClosureLikes, "inline calls to locally defined closures"
        )

        # pre typing
        if not state.flags.no_rewrites:
            pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
            pm.add_pass(DeadBranchPrune, "dead branch pruning")
            pm.add_pass(GenericRewrites, "nopython rewrites")

        # convert any remaining closures into functions
        pm.add_pass(
            MakeFunctionToJitFunction,
            "convert make_function into JIT functions",
        )
        # inline functions that have been determined as inlinable and rerun
        # branch pruning, this needs to be run after closures are inlined as
        # the IR repr of a closure masks call sites if an inlinable is called
        # inside a closure
        pm.add_pass(InlineInlinables, "inline inlinable functions")
        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, "dead branch pruning")

        pm.add_pass(FindLiterallyCalls, "find literally calls")
        pm.add_pass(LiteralUnroll, "handles literal_unroll")

        if state.flags.enable_ssa:
            pm.add_pass(ReconstructSSA, "ssa")

        pm.add_pass(LiteralPropagationSubPipelinePass, "Literal propagation")

        pm.finalize()
        return pm

    @staticmethod
    def define_typed_pipeline(state, name="dpex_offload_typed"):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        # Annotate only once legalized
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(
            RewriteNdarrayFunctionsPass,
            "Rewrite numpy.ndarray functions to dpnp.ndarray functions",
        )

        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimization
        pm.add_pass(InlineOverloads, "inline overloaded functions")
        pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        pm.add_pass(ParforPass, "convert to parfors")

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name="dpex_offload_lowering"):
        """Returns an nopython mode pipeline based PassManager"""
        pm = PassManager(name)

        # legalize
        pm.add_pass(
            NoPythonSupportedFeatureValidation,
            "ensure features that are in use are in a valid form",
        )
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(DpexLowering, "Custom Lowerer with auto-offload support")
        pm.add_pass(NoPythonBackend, "nopython mode backend")
        pm.add_pass(DumpParforDiagnostics, "dump parfor diagnostics")

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_pipeline(state, name="dpex_offload_nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        # compose pipeline from untyped, typed and lowering parts
        dpb = _OffloadPassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm


class OffloadCompiler(CompilerBase):
    """Dpex's offload compiler pipeline that can offload parfor nodes and
    other NumPy-based library calls to a SYCL device.

    .. note:: Deprecated in 0.20
          The offload compiler is deprecated and will be removed in a future
          release.
    """

    def define_pipelines(self):
        pms = []
        self.state.parfor_diagnostics = ExtendedParforDiagnostics()
        self.state.metadata[
            "parfor_diagnostics"
        ] = self.state.parfor_diagnostics
        if not self.state.flags.force_pyobject:
            pms.append(_OffloadPassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            raise UnsupportedCompilationModeError()
        return pms
