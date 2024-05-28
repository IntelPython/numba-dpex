# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings

from numba.core import ir, types
from numba.core.errors import NumbaParallelSafetyWarning
from numba.core.ir_utils import (
    get_name_var_table,
    get_unused_var_name,
    legalize_names,
    mk_unique_var,
    remove_dels,
    replace_var_names,
)
from numba.core.typing import signature
from numba.parfors import parfor

from numba_dpex.core import config
from numba_dpex.core.decorators import kernel
from numba_dpex.core.parfors.parfor_sentinel_replace_pass import (
    ParforBodyArguments,
)
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType
from numba_dpex.core.utils.call_kernel_builder import SPIRVKernelModule
from numba_dpex.kernel_api_impl.spirv.dispatcher import (
    SPIRVKernelDispatcher,
    _SPIRVKernelCompileResult,
)

from .kernel_templates.range_kernel_template import RangeKernelTemplate


class ParforKernel:
    def __init__(
        self,
        signature,
        kernel_args,
        kernel_arg_types,
        local_accessors=None,
        work_group_size=None,
        kernel_module=None,
    ):
        self.signature = signature
        self.kernel_args = kernel_args
        self.kernel_arg_types = kernel_arg_types
        self.local_accessors = local_accessors
        self.work_group_size = work_group_size
        self.kernel_module = kernel_module


def _legalize_names_with_typemap(names, typemap):
    """Replace illegal characters in Numba IR var names.

    We use ir_utils.legalize_names to replace internal IR variable names
    containing illegal characters (e.g. period) with a legal character
    (underscore) so as to create legal variable names. The original variable
    names are in the typemap so we also need to add the legalized name to the
    typemap as well.
    """
    outdict = legalize_names(names)
    # For each pair in the dict of legalized names...
    for x, y in outdict.items():
        # If the name had some legalization change to it...
        if x != y:
            # Set the type of the new name the same as the type of the old name.
            typemap[y] = typemap[x]
    return outdict


def _to_scalar_from_0d(x):
    if isinstance(x, types.ArrayCompatible) and x.ndim == 0:
        return x.dtype
    else:
        return x


def _replace_var_with_array_in_block(vars, block, typemap, calltypes):
    new_block = []
    for inst in block.body:
        if isinstance(inst, ir.Assign) and inst.target.name in vars:
            const_node = ir.Const(0, inst.loc)
            const_var = ir.Var(
                inst.target.scope, mk_unique_var("$const_ind_0"), inst.loc
            )
            typemap[const_var.name] = types.uintp
            const_assign = ir.Assign(const_node, const_var, inst.loc)
            new_block.append(const_assign)

            setitem_node = ir.SetItem(
                inst.target, const_var, inst.value, inst.loc
            )
            calltypes[setitem_node] = signature(
                types.none,
                types.npytypes.Array(typemap[inst.target.name], 1, "C"),
                types.intp,
                typemap[inst.target.name],
            )
            new_block.append(setitem_node)
            continue
        elif isinstance(inst, parfor.Parfor):
            _replace_var_with_array_internal(
                vars, {0: inst.init_block}, typemap, calltypes
            )
            _replace_var_with_array_internal(
                vars, inst.loop_body, typemap, calltypes
            )

        new_block.append(inst)
    return new_block


def _replace_var_with_array_internal(vars, loop_body, typemap, calltypes):
    for label, block in loop_body.items():
        block.body = _replace_var_with_array_in_block(
            vars, block, typemap, calltypes
        )


def _replace_var_with_array(vars, loop_body, typemap, calltypes):
    _replace_var_with_array_internal(vars, loop_body, typemap, calltypes)
    for v in vars:
        el_typ = typemap[v]
        typemap.pop(v, None)
        typemap[v] = types.npytypes.Array(el_typ, 1, "C")


def create_kernel_for_parfor(
    lowerer,
    parfor_node,
    typemap,
    loop_ranges,
    races,
    parfor_outputs,
) -> ParforKernel:
    """
    Creates a numba_dpex.kernel function for a parfor node.

    There are two parts to this function:

        1) Code to iterate across the iteration space as defined by
           the schedule.
        2) The parfor body that does the work for a single point in
           the iteration space.

    Part 1 is created as Python text for simplicity with a sentinel
    assignment to mark the point in the IR where the parfor body
    should be added. This Python text is 'exec'ed into existence and its
    IR retrieved with run_frontend. The IR is scanned for the sentinel
    assignment where that basic block is split and the IR for the parfor
    body inserted.
    """
    loc = parfor_node.init_block.loc

    # The parfor body and the main function body share ir.Var nodes.
    # We have to do some replacements of Var names in the parfor body
    # to make them legal parameter names. If we don't copy then the
    # Vars in the main function also would incorrectly change their name.
    loop_body = copy.copy(parfor_node.loop_body)
    remove_dels(loop_body)

    parfor_dim = len(parfor_node.loop_nests)
    loop_indices = [
        loop_nest.index_variable.name for loop_nest in parfor_node.loop_nests
    ]

    # Get all the parfor params.
    parfor_params = parfor_node.params
    typemap = lowerer.fndesc.typemap

    # Compute just the parfor inputs as a set difference.
    parfor_inputs = sorted(list(set(parfor_params) - set(parfor_outputs)))

    for race in races:
        msg = (
            "Variable %s used in parallel loop may be written "
            "to simultaneously by multiple workers and may result "
            "in non-deterministic or unintended results." % race
        )
        warnings.warn(NumbaParallelSafetyWarning(msg, loc))

    _replace_var_with_array(races, loop_body, typemap, lowerer.fndesc.calltypes)

    # Reorder all the params so that inputs go first then outputs.
    parfor_params = parfor_inputs + parfor_outputs

    # Some Var and loop_indices may not have legal parameter names so create a
    # dict of potentially illegal param name to guaranteed legal name.
    param_dict = _legalize_names_with_typemap(parfor_params, typemap)
    ind_dict = _legalize_names_with_typemap(loop_indices, typemap)

    # Compute a new list of legal loop index names.
    legal_loop_indices = [ind_dict[v] for v in loop_indices]
    # Get the types of each parameter.
    param_types = [_to_scalar_from_0d(typemap[v]) for v in parfor_params]
    # Calculate types of args passed to the kernel function.
    func_arg_types = [typemap[v] for v in (parfor_inputs + parfor_outputs)]

    # Replace illegal parameter names in the loop body with legal ones.
    replace_var_names(loop_body, param_dict)

    # remember the name before legalizing as the actual arguments
    parfor_args = parfor_params
    # Change parfor_params to be legal names.
    parfor_params = [param_dict[v] for v in parfor_params]
    parfor_params_orig = parfor_params

    parfor_params = []
    ascontig = False
    for pindex in range(len(parfor_params_orig)):
        if (
            ascontig
            and pindex < len(parfor_inputs)
            and isinstance(param_types[pindex], types.npytypes.Array)
        ):
            parfor_params.append(parfor_params_orig[pindex] + "param")
        else:
            parfor_params.append(parfor_params_orig[pindex])

    # Change parfor body to replace illegal loop index vars with legal ones.
    replace_var_names(loop_body, ind_dict)

    loop_body_var_table = get_name_var_table(loop_body)
    sentinel_name = get_unused_var_name("__sentinel__", loop_body_var_table)

    if config.DEBUG_ARRAY_OPT >= 1:
        print("legal parfor_params = ", parfor_params, type(parfor_params))

    # Determine the unique names of the kernel functions.
    kernel_name = "__dpex_parfor_kernel_%s" % (parfor_node.id)

    kernel_template = RangeKernelTemplate(
        kernel_name=kernel_name,
        kernel_params=parfor_params,
        kernel_rank=parfor_dim,
        ivar_names=legal_loop_indices,
        sentinel_name=sentinel_name,
        loop_ranges=loop_ranges,
        param_dict=param_dict,
    )

    kernel_dispatcher: SPIRVKernelDispatcher = kernel(
        kernel_template.py_func,
        _parfor_body_args=ParforBodyArguments(
            loop_body=loop_body,
            param_dict=param_dict,
            legal_loop_indices=legal_loop_indices,
        ),
    )

    # The first argument to a range kernel is a kernel_api.NdItem object. The
    # ``NdItem`` object is used by the kernel_api.spirv backend to generate the
    # correct SPIR-V indexing instructions. Since, the argument is not something
    # available originally in the kernel_param_types, we add it at this point to
    # make sure the kernel signature matches the actual generated code.
    ty_item = ItemType(parfor_dim)
    kernel_param_types = (ty_item, *param_types)
    kernel_sig = signature(types.none, *kernel_param_types)

    kcres: _SPIRVKernelCompileResult = kernel_dispatcher.get_compile_result(
        types.void(*kernel_param_types)  # kernel signature
    )
    kernel_module: SPIRVKernelModule = kcres.kernel_device_ir_module

    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return ParforKernel(
        signature=kernel_sig,
        kernel_args=parfor_args,
        kernel_arg_types=func_arg_types,
        kernel_module=kernel_module,
    )


def update_sentinel(kernel_ir, sentinel_name, kernel_body, new_label):
    """Searched all the blocks in the IR generated from a kernel template and
    replaces the __sentinel__ instruction with the actual op for the parfor.

    Args:
        kernel_ir : Numba FunctionIR that was generated from a kernel template
        sentinel_name : The name of the sentinel instruction that is to be
        replaced.
        kernel_body : The function body of the kernel template generated
        numba_dpex.kernel function
        new_label: The new label to be used for the basic block created to store
        the instructions that replaced the sentinel
    """
    for label, block in kernel_ir.blocks.items():
        for i, inst in enumerate(block.body):
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name == sentinel_name
            ):
                # We found the sentinel assignment.
                loc = inst.loc
                scope = block.scope
                # split block across __sentinel__
                # A new block is allocated for the statements prior to the
                # sentinel but the new block maintains the current block label.
                prev_block = ir.Block(scope, loc)
                prev_block.body = block.body[:i]

                # The current block is used for statements after the sentinel.
                block.body = block.body[i + 1 :]  # noqa: E203
                # But the current block gets a new label.
                body_first_label = min(kernel_body.keys())

                # The previous block jumps to the minimum labelled block of the
                # parfor body.
                prev_block.append(ir.Jump(body_first_label, loc))

                # Add all the parfor loop body blocks to the kernel IR
                for loop, b in kernel_body.items():
                    kernel_ir.blocks[loop] = copy.copy(b)
                    kernel_ir.blocks[loop].body = copy.copy(
                        kernel_ir.blocks[loop].body
                    )

                body_last_label = max(kernel_body.keys())
                kernel_ir.blocks[new_label] = block
                kernel_ir.blocks[label] = prev_block
                # Add a jump from the last parfor body block to the block
                # containing statements after the sentinel.
                kernel_ir.blocks[body_last_label].append(
                    ir.Jump(new_label, loc)
                )

                break
        else:
            continue
        break
