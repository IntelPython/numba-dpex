# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.compiler import CompilerBase
from numba.core.compiler_machinery import PassManager
from numba.core.typed_passes import (
    AnnotateTypes,
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

from numba_dpex.core import config
from numba_dpex.core.exceptions import UnsupportedCompilationModeError
from numba_dpex.core.passes.passes import (
    ConstantSizeStaticLocalMemoryPass,
    NoPythonBackend,
    QualNameDisambiguationLowering,
)


class _KernelPassBuilder(object):
    """
    A pass builder to run dpex's code-generation and optimization passes.

    Unlike Numba, dpex's pass builder does not offer objectmode and
    interpreted passes.
    """

    @staticmethod
    def define_untyped_pipeline(state, name="dpex_kernel_untyped"):
        """Returns an untyped part of the nopython pipeline

        The pipeline of untyped passes is duplicated from Numba's compiler. We
        are adding couple of passes to the pipeline to change specific numpy
        overloads.
        """
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # --- dpex passes added to the untyped pipeline                      --#

        # Add pass to ensure when users allocate static constant memory the
        # size of the allocation is a constant and not specified by a closure
        # variable.
        if config.STATIC_LOCAL_MEM_PASS:
            pm.add_pass(
                ConstantSizeStaticLocalMemoryPass,
                "dpex constant size for static local memory",
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
    def define_typed_pipeline(state, name="dpex_kernel_typed"):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        # Annotate only once legalized
        pm.add_pass(AnnotateTypes, "annotate types")
        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimization
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name="dpex_kernel_lowering"):
        """Returns an nopython mode pipeline based PassManager"""
        pm = PassManager(name)

        # legalize
        pm.add_pass(
            NoPythonSupportedFeatureValidation,
            "ensure features that are in use are in a valid form",
        )
        pm.add_pass(IRLegalization, "ensure IR is legal prior to lowering")

        # NativeLowering has some issue with freevar ambiguity,
        # therefore, we are using QualNameDisambiguationLowering instead
        # numba-dpex github issue: https://github.com/IntelPython/numba-dpex/issues/898
        pm.add_pass(
            QualNameDisambiguationLowering,
            "numba_dpex qualified name disambiguation",
        )
        pm.add_pass(NoPythonBackend, "nopython mode backend")

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_pipeline(state, name="dpex_kernel_nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        # compose pipeline from untyped, typed and lowering parts
        dpb = _KernelPassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm


class KernelCompiler(CompilerBase):
    """Dpex's kernel compilation pipeline."""

    def define_pipelines(self):
        pms = []
        if not self.state.flags.force_pyobject:
            pms.append(_KernelPassBuilder.define_nopython_pipeline(self.state))
        if self.state.status.can_fallback or self.state.flags.force_pyobject:
            raise UnsupportedCompilationModeError()

        # Compile the kernel without generating a cpython or a cfunc wrapper
        self.state.flags.no_cpython_wrapper = True
        self.state.flags.no_cfunc_wrapper = True
        # The pass pipeline does not generate an executable when compiling a
        # kernel function. Instead, the
        # kernel_dispatcher._KernelCompiler.compile generates the executable in
        # the form of a host callable launcher function
        self.state.flags.no_compile = True

        return pms
