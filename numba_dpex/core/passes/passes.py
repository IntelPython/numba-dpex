# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

from numba.core import errors, ir, typing
from numba.core.compiler_machinery import (
    AnalysisPass,
    FunctionPass,
    register_pass,
)
from numba.core.ir_utils import remove_dels
from numba.core.typed_passes import NativeLowering

from numba_dpex.core import config


@register_pass(mutates_CFG=True, analysis_only=False)
class NoPythonBackend(FunctionPass):
    _name = "dpex_nopython_backend"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """

        lowered = state["cr"]
        signature = typing.signature(state.return_type, *state.args)

        from numba.core.compiler import compile_result

        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            entry_point=lowered.cfunc,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=False,
            lifted=state.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=state.metadata,
            reload_init=state.reload_init,
        )

        remove_dels(state.func_ir.blocks)

        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class DumpParforDiagnostics(AnalysisPass):
    _name = "dpex_dump_parfor_diagnostics"

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        # if state.flags.auto_parallel.enabled, add a condition flag for kernels
        if config.OFFLOAD_DIAGNOSTICS:
            if state.parfor_diagnostics is not None:
                state.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
            else:
                raise RuntimeError("Diagnostics failed.")
        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class QualNameDisambiguationLowering(NativeLowering):
    """Qualified name disambiguation lowering pass

    If there are multiple @func decorated functions exist inside
    another @func decorated block, the numba compiler machinery
    creates same qualified names for different compiled function.
    Therefore, we utilize `unique_name` to resolve the ambiguity.

    Args:
        NativeLowering (CompilerPass): Superclass from which this
        class has been inherited.

    Returns:
        bool: True if `run_pass()` of the superclass is successful.
    """

    _name = "qual-name-disambiguation-lowering"

    def run_pass(self, state):
        qual_name = state.func_id.func_qualname
        state.func_id.func_qualname = state.func_id.unique_name
        ret = NativeLowering.run_pass(self, state)
        state.func_id.func_qualname = qual_name
        return ret
