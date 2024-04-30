# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import PassManager
from numba.core.typed_passes import (
    IRLegalization,
    NoPythonSupportedFeatureValidation,
)
from numba.core.untyped_passes import InlineClosureLikes

from numba_dpex.core.exceptions import UnsupportedCompilationModeError
from numba_dpex.core.parfors.parfor_sentinel_replace_pass import (
    ParforSentinelReplacePass,
)
from numba_dpex.core.passes.passes import (
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
        kdpb = _KernelPassBuilder
        ndpb = DefaultPassBuilder
        pm = PassManager(name)
        untyped_passes = ndpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)
        # TODO: create separate parfor kernel pass
        pm.add_pass_after(ParforSentinelReplacePass, InlineClosureLikes)

        typed_passes = ndpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = kdpb.define_nopython_lowering_pipeline(state)
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
