# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
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

from numba_dpex import config


@register_pass(mutates_CFG=True, analysis_only=False)
class ConstantSizeStaticLocalMemoryPass(FunctionPass):
    _name = "dpex_constant_size_static_local_memory_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        func_ir = state.func_ir

        work_list = list(func_ir.blocks.items())
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr):
                        if expr.op == "call":
                            find_var = block.find_variable_assignment(
                                expr.func.name
                            )
                            if find_var is not None:
                                call_node = find_var.value
                                if (
                                    isinstance(call_node, ir.Expr)
                                    and call_node.op == "getattr"
                                    and call_node.attr == "array"
                                ):
                                    # let's check if it is from numba_dpex.local
                                    attr_node = block.find_variable_assignment(
                                        call_node.value.name
                                    ).value
                                    if (
                                        isinstance(attr_node, ir.Expr)
                                        and attr_node.op == "getattr"
                                        and attr_node.attr == "local"
                                    ):
                                        arg = None
                                        # at first look in keyword arguments to
                                        # get the shape, which has to be
                                        # constant
                                        if expr.kws:
                                            for _arg in expr.kws:
                                                if _arg[0] == "shape":
                                                    arg = _arg[1]

                                        if not arg:
                                            arg = expr.args[0]

                                        error = False
                                        # arg can be one constant or a tuple of
                                        # constant items
                                        arg_type = func_ir.get_definition(
                                            arg.name
                                        )
                                        if isinstance(arg_type, ir.Expr):
                                            # we have a tuple
                                            for item in arg_type.items:
                                                if not isinstance(
                                                    func_ir.get_definition(
                                                        item.name
                                                    ),
                                                    ir.Const,
                                                ):
                                                    error = True
                                                    break

                                        else:
                                            if not isinstance(
                                                func_ir.get_definition(
                                                    arg.name
                                                ),
                                                ir.Const,
                                            ):
                                                error = True
                                                break

                                        if error:
                                            warnings.warn_explicit(
                                                "The size of the Local memory "
                                                + "has to be constant",
                                                errors.NumbaError,
                                                state.func_id.filename,
                                                state.func_id.firstlineno,
                                            )
                                            raise

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


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
