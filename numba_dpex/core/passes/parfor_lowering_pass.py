# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy

from numba.core import funcdesc, ir
from numba.core.compiler_machinery import LoweringPass, register_pass
from numba.core.lowering import Lower
from numba.parfors.parfor_lowering import (
    _lower_parfor_parallel as _lower_parfor_parallel_std,
)

from numba_dpex import config

from ..exceptions import UnsupportedParforError
from ..utils.kernel_builder import create_kernel_for_parfor
from .parfor import Parfor, find_potential_aliases_parfor, get_parfor_outputs


def _lower_parfor_gufunc(lowerer, parfor):
    """Lowers a parfor node created by the dpjit compiler to a kernel.

    The general approach is as follows:

        - The code from the parfor's init block is lowered normally
        in the context of the current function.
        - The body of the parfor is transformed into a gufunc function.

    """
    typingctx = lowerer.context.typing_context
    targetctx = lowerer.context

    # We copy the typemap here because for race condition variable we'll
    # update their type to array so they can be updated by the gufunc.
    orig_typemap = lowerer.fndesc.typemap

    # replace original typemap with copy and restore the original at the end.
    lowerer.fndesc.typemap = copy.copy(orig_typemap)

    if config.DEBUG_ARRAY_OPT:
        print("lowerer.fndesc", lowerer.fndesc, type(lowerer.fndesc))

    typemap = lowerer.fndesc.typemap
    varmap = lowerer.varmap

    loc = parfor.init_block.loc
    scope = parfor.init_block.scope

    for instr in parfor.init_block.body:
        lowerer.lower_inst(instr)

    for racevar in parfor.races:
        if racevar not in varmap:
            rvtyp = typemap[racevar]
            rv = ir.Var(scope, racevar, loc)
            lowerer._alloca_var(rv.name, rvtyp)

    alias_map = {}
    arg_aliases = {}

    find_potential_aliases_parfor(
        parfor, parfor.params, typemap, lowerer.func_ir, alias_map, arg_aliases
    )

    # run get_parfor_outputs() and get_parfor_reductions() before
    # gufunc creation since Jumps are modified so CFG of loop_body
    # dict will become invalid
    if parfor.params is None:
        raise AssertionError

    parfor_output_arrays = get_parfor_outputs(parfor, parfor.params)

    # compile parfor body as a separate function to be used with GUFuncWrapper
    flags = copy.copy(parfor.flags)
    flags.error_model = "numpy"

    # Can't get here unless flags.set('auto_parallel', ParallelOptions(True))
    index_var_typ = typemap[parfor.loop_nests[0].index_variable.name]

    # index variables should have the same type, check rest of indices
    for loop_nest in parfor.loop_nests[1:]:
        if typemap[loop_nest.index_variable.name] != index_var_typ:
            raise AssertionError

    loop_ranges = [
        (loop_nest.start, loop_nest.stop, loop_nest.step)
        for loop_nest in parfor.loop_nests
    ]

    try:
        (
            func,
            func_args,
            func_sig,
            func_arg_types,
            modified_arrays,
        ) = create_kernel_for_parfor(
            lowerer,
            parfor,
            typemap,
            flags,
            loop_ranges,
            bool(alias_map),
            parfor.races,
            parfor_output_arrays,
        )
    except Exception:
        # FIXME: Make the exception more informative
        raise UnsupportedParforError

    num_inputs = len(func_args) - len(parfor_output_arrays)
    if config.DEBUG_ARRAY_OPT:
        print("func", func, type(func))
        print("func_args", func_args, type(func_args))
        print("func_sig", func_sig, type(func_sig))
        print("num_inputs = ", num_inputs)
        print("parfor_outputs = ", parfor_output_arrays)

    # call the func in parallel by wrapping it with ParallelGUFuncBuilder
    if config.DEBUG_ARRAY_OPT:
        print("loop_nests = ", parfor.loop_nests)
        print("loop_ranges = ", loop_ranges)

    # gu_signature = _create_shape_signature(
    #     parfor.get_shape_classes,
    #     num_inputs,
    #     func_args,
    #     func_sig,
    #     parfor.races,
    #     typemap,
    # )

    # generate_kernel_launch_ops(
    #     lowerer,
    #     func,
    #     gu_signature,
    #     func_sig,
    #     func_args,
    #     num_inputs,
    #     func_arg_types,
    #     loop_ranges,
    #     modified_arrays,
    # )

    # Restore the original typemap of the function that was replaced
    # temporarily at the beginning of this function.
    lowerer.fndesc.typemap = orig_typemap


class _ParforLower(Lower):
    """Extends standard lowering to accommodate parfor.Parfor nodes that may
    have the `lowerer` attribute set.
    """

    # custom instruction lowering to handle parfor nodes
    def lower_inst(self, inst):
        if isinstance(inst, Parfor):
            # FIXME: Temporary for testing
            # inst.lowerer = _lower_parfor_gufunc

            if inst.lowerer is None:
                _lower_parfor_parallel_std(self, inst)
            else:
                inst.lowerer(self, inst)
        else:
            super().lower_inst(inst)


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforLoweringPass(LoweringPass):
    """A custom lowering pass that does dpex-specific lowering of parfor
    nodes.

    FIXME: Redesign once numba-dpex supports Numba 0.57
    """

    _name = "dpex_parfor_lowering"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        targetctx = state.targetctx

        library = state.library
        interp = state.func_ir
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata

        kwargs = {}

        # for support numba 0.54 and <=0.55.0dev0=*_469
        if hasattr(flags, "get_mangle_string"):
            kwargs["abi_tags"] = flags.get_mangle_string()
        # Lowering
        fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(
            interp,
            typemap,
            restype,
            calltypes,
            mangler=targetctx.mangler,
            inline=flags.forceinline,
            noalias=flags.noalias,
            **kwargs,
        )

        with targetctx.push_code_library(library):
            lower = _ParforLower(
                targetctx, library, fndesc, interp, metadata=metadata
            )
            lower.lower()
            if not flags.no_cpython_wrapper:
                lower.create_cpython_wrapper(flags.release_gil)

            env = lower.env
            call_helper = lower.call_helper
            del lower

        from numba.core.compiler import _LowerResult  # TODO: move this

        if flags.no_compile:
            state["cr"] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
        else:
            # Prepare for execution
            cfunc = targetctx.get_executable(library, fndesc, env)
            # Insert native function for use by other jitted-functions.
            # We also register its library to allow for inlining.
            targetctx.insert_user_function(cfunc, fndesc, [library])
            state["cr"] = _LowerResult(
                fndesc, call_helper, cfunc=cfunc, env=env
            )

        return True
