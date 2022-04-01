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
    LiteralUnroll,
    MakeFunctionToJitFunction,
    ReconstructSSA,
    RewriteSemanticConstants,
    TranslateByteCode,
    WithLifting,
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
    RewriteNdarrayFunctionsPass,
    RewriteOverloadedNumPyFunctionsPass,
)


class PassBuilder(object):
    """
    This is a pass builder to run Intel GPU/CPU specific
    code-generation and optimization passes. This pass builder does
    not offer objectmode and interpreted passes.
    """

    @staticmethod
    def default_numba_nopython_pipeline(state, pm):
        """Adds the default set of NUMBA passes to the pass manager"""
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, "analyzing bytecode")
            pm.add_pass(FixupArgs, "fix up args")
        pm.add_pass(IRProcessing, "processing IR")
        pm.add_pass(WithLifting, "Handle with contexts")

        # this pass rewrites name of NumPy functions we intend to overload
        pm.add_pass(
            RewriteOverloadedNumPyFunctionsPass,
            "Rewrite name of Numpy functions to overload already overloaded function",
        )

        # Add pass to ensure when users are allocating static
        # constant memory the size is a constant and can not
        # come from a closure variable
        pm.add_pass(
            ConstantSizeStaticLocalMemoryPass,
            "dpex constant size for static local memory",
        )

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

        # typing
        pm.add_pass(NopythonTypeInference, "nopython frontend")
        pm.add_pass(AnnotateTypes, "annotate types")

        pm.add_pass(
            RewriteNdarrayFunctionsPass,
            "Rewrite numpy.ndarray functions to dpnp.ndarray functions",
        )

        # strip phis
        pm.add_pass(PreLowerStripPhis, "remove phis nodes")

        # optimisation
        pm.add_pass(InlineOverloads, "inline overloaded functions")

    @staticmethod
    def define_nopython_pipeline(state, name="dpex_nopython"):
        """Returns an nopython mode pipeline based PassManager"""
        pm = PassManager(name)
        PassBuilder.default_numba_nopython_pipeline(state, pm)

        # Intel GPU/CPU specific optimizations
        pm.add_pass(PreParforPass, "Preprocessing for parfors")
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, "nopython rewrites")
        pm.add_pass(ParforPass, "convert to parfors")

        # legalise
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
