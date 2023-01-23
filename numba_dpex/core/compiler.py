# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from types import FunctionType

from numba.core import compiler, ir
from numba.core import types as numba_types
from numba.core.compiler import CompilerBase
from numba.core.compiler_lock import global_compiler_lock
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

from numba_dpex import config
from numba_dpex.core.exceptions import (
    KernelHasReturnValueError,
    UnreachableError,
    UnsupportedCompilationModeError,
)
from numba_dpex.core.passes.passes import (
    ConstantSizeStaticLocalMemoryPass,
    DpexLowering,
    DumpParforDiagnostics,
    NoPythonBackend,
    ParforPass,
    PreParforPass,
)
from numba_dpex.core.passes.rename_numpy_functions_pass import (
    IdentifyNumPyFunctionsPass,
    RewriteNdarrayFunctionsPass,
    RewriteOverloadedNumPyFunctionsPass,
)
from numba_dpex.parfor_diagnostics import ExtendedParforDiagnostics


class PassBuilder(object):
    """
    A pass builder to run dpex's code-generation and optimization passes.

    Unlike Numba, dpex's pass builder does not offer objectmode and
    interpreted passes.
    """

    @staticmethod
    def define_untyped_pipeline(state, name="dpex_untyped"):
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

        # --- Begin dpex passes added to the untyped pipeline                --#

        # The RewriteOverloadedNumPyFunctionsPass rewrites the module namespace
        # of specific NumPy functions to dpnp, as we overload these functions
        # differently.
        pm.add_pass(
            RewriteOverloadedNumPyFunctionsPass,
            "Rewrite name of Numpy functions to overload already overloaded "
            + "function",
        )
        # this pass count number of Numpy functions calls
        pm.add_pass(
            IdentifyNumPyFunctionsPass,
            "Identify number of NumPy functions Calls",
        )
        # Add pass to ensure when users allocate static constant memory the
        # size of the allocation is a constant and not specified by a closure
        # variable.
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
    def define_typed_pipeline(state, name="dpex_typed"):
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
    def define_nopython_lowering_pipeline(state, name="dpex_nopython_lowering"):
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
    def define_nopython_pipeline(state, name="dpex_nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        # compose pipeline from untyped, typed and lowering parts
        dpb = PassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm


class Compiler(CompilerBase):
    """Dpex's compiler pipeline."""

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
            raise UnsupportedCompilationModeError()
        return pms


@global_compiler_lock
def compile_with_dpex(
    pyfunc,
    pyfunc_name,
    args,
    return_type,
    target_context,
    typing_context,
    debug=False,
    is_kernel=True,
    extra_compile_flags=None,
):
    """
    Compiles a function using the dpex compiler pipeline and returns the
    compiled result.

    Args:
        args: The list of arguments passed to the kernel.
        debug (bool): Optional flag to turn on debug mode compilation.
        extra_compile_flags: Extra flags passed to the compiler.

    Returns:
        cres: Compiled result.

    Raises:
        KernelHasReturnValueError: If the compiled function returns a
        non-void value.
    """
    # First compilation will trigger the initialization of the backend.
    typingctx = typing_context
    targetctx = target_context

    flags = compiler.Flags()
    # Do not compile the function to a binary, just lower to LLVM
    flags.debuginfo = config.DEBUGINFO_DEFAULT
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.nrt = False

    if debug:
        flags.debuginfo = debug

    # Run compilation pipeline
    if isinstance(pyfunc, FunctionType):
        cres = compiler.compile_extra(
            typingctx=typingctx,
            targetctx=targetctx,
            func=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )
    elif isinstance(pyfunc, ir.FunctionIR):
        cres = compiler.compile_ir(
            typingctx=typingctx,
            targetctx=targetctx,
            func_ir=pyfunc,
            args=args,
            return_type=return_type,
            flags=flags,
            locals={},
            pipeline_class=Compiler,
        )
    else:
        raise UnreachableError()

    if (
        is_kernel
        and cres.signature.return_type is not None
        and cres.signature.return_type != numba_types.void
    ):
        raise KernelHasReturnValueError(
            kernel_name=pyfunc_name,
            return_type=cres.signature.return_type,
        )
    # Linking depending libraries
    library = cres.library
    library.finalize()

    return cres
