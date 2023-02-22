# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import copy
import sys
import warnings

import dpctl.program as dpctl_prog
import dpnp
import numba
from numba.core import compiler, ir, types
from numba.core.errors import NumbaParallelSafetyWarning
from numba.core.ir_utils import (
    add_offset_to_labels,
    get_name_var_table,
    get_unused_var_name,
    legalize_names,
    mk_unique_var,
    remove_dead,
    remove_dels,
    rename_labels,
    replace_var_names,
)
from numba.core.typing import signature

import numba_dpex as dpex
from numba_dpex import config
from numba_dpex.utils import address_space

from ..descriptor import dpex_kernel_target
from ..kernel_interface.utils import determine_kernel_launch_queue
from ..passes import parfor
from ..types.dpnp_ndarray_type import DpnpNdArray


def _print_block(block):
    for i, inst in enumerate(block.body):
        print("    ", i, inst)


def _print_body(body_dict):
    """Pretty-print a set of IR blocks."""
    for label, block in body_dict.items():
        print("label: ", label)
        _print_block(block)


def _compile_kernel_parfor(sycl_queue, kernel_name, func_ir, args, debug=False):
    # Create a SPIRVKernel object
    kernel = dpex.core.kernel_interface.spirv_kernel.SpirvKernel(
        func_ir, kernel_name
    )

    # compile the kernel
    kernel.compile(
        args=args,
        typing_ctx=dpex_kernel_target.typing_context,
        target_ctx=dpex_kernel_target.target_context,
        debug=debug,
        compile_flags=None,
    )

    # Compile a SYCL Kernel object rom the SPIRVKernel

    dpctl_create_program_from_spirv_flags = []

    if debug or config.OPT == 0:
        # if debug is ON we need to pass additional flags to igc.
        dpctl_create_program_from_spirv_flags = ["-g", "-cl-opt-disable"]

    # create a program
    kernel_bundle = dpctl_prog.create_program_from_spirv(
        sycl_queue,
        kernel.device_driver_ir_module,
        " ".join(dpctl_create_program_from_spirv_flags),
    )
    #  create a kernel
    sycl_kernel = kernel_bundle.get_sycl_kernel(kernel.module_name)

    return sycl_kernel


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


def _get_param_addrspace(
    params, typemap, default_addrspace=address_space.GLOBAL
):
    addrspaces = []
    for param in params:
        if isinstance(_to_scalar_from_0d(typemap[param]), DpnpNdArray):
            addrspaces.append(default_addrspace)
        else:
            addrspaces.append(None)
    return addrspaces


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


def _dbgprint_after_each_array_assignments(lowerer, loop_body, typemap):
    for label, block in loop_body.items():
        new_block = block.copy()
        new_block.clear()
        loc = block.loc
        scope = block.scope
        for inst in block.body:
            new_block.append(inst)
            # Append print after assignment
            if isinstance(inst, ir.Assign):
                # Only apply to numbers
                if typemap[inst.target.name] not in types.number_domain:
                    continue

                # Make constant string
                strval = "{} =".format(inst.target.name)
                strconsttyp = types.StringLiteral(strval)

                lhs = ir.Var(scope, mk_unique_var("str_const"), loc)
                assign_lhs = ir.Assign(
                    value=ir.Const(value=strval, loc=loc), target=lhs, loc=loc
                )
                typemap[lhs.name] = strconsttyp
                new_block.append(assign_lhs)

                # Make print node
                print_node = ir.Print(
                    args=[lhs, inst.target], vararg=None, loc=loc
                )
                new_block.append(print_node)
                sig = signature(
                    types.none, typemap[lhs.name], typemap[inst.target.name]
                )
                lowerer.fndesc.calltypes[print_node] = sig
        loop_body[label] = new_block


def _generate_kernel_indexing(
    parfor_dim, legal_loop_indices, loop_ranges, param_dict, has_reduction, redvars, typemap
):
    """Generates the indexing intrinsic calls inside a dpex kernel.

    Args:
        parfor_dim (_type_): _description_
        legal_loop_indices (_type_): _description_
        loop_ranges (_type_): _description_
        param_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    gufunc_txt = ""
    global_id_dim = 0
    for_loop_dim = parfor_dim

    if parfor_dim > 3:
        raise NotImplementedError
        global_id_dim = 3
    else:
        global_id_dim = parfor_dim

    for eachdim in range(global_id_dim):
        gufunc_txt += (
            "    "
            + legal_loop_indices[eachdim]
            + " = "
            + "dpex.get_global_id("
            + str(eachdim)
            + ")\n"
        )

    if has_reduction:
        for eachdim in range(global_id_dim):
            gufunc_txt += (
                "    "
                + f"local_id{eachdim} = "
                + "dpex.get_local_id("
                + str(eachdim)
                + ")\n"
            )
        for eachdim in range(global_id_dim):
            gufunc_txt += (
                "    "
                + f"local_size{eachdim} = "
                + "dpex.get_local_size("
                + str(eachdim)
                + ")\n"
            )
        for eachdim in range(global_id_dim):
            gufunc_txt += (
                "    "
                + f"group_id{eachdim} = "
                + "dpex.get_local_size("
                + str(eachdim)
                + ")\n"
            )

        # Allocate local_sums arrays for each reduction variable.
        for redvar in redvars:
            rtyp = str(typemap[redvar])
            gufunc_txt += (
                "    "
                + f"local_sums_{redvar} = "
                + f"dpex.local.array(8, {rtyp})\n"
            )
            gufunc_txt += (
                "    "
                + f"local_sums_{redvar}[local_id0] = 0\n"
            )

    for eachdim in range(global_id_dim, for_loop_dim):
        for indent in range(1 + (eachdim - global_id_dim)):
            gufunc_txt += "    "

        start, stop, step = loop_ranges[eachdim]
        start = param_dict.get(str(start), start)
        stop = param_dict.get(str(stop), stop)
        gufunc_txt += (
            "for "
            + legal_loop_indices[eachdim]
            + " in range("
            + str(start)
            + ", "
            + str(stop)
            + " + 1):\n"
        )

    for eachdim in range(global_id_dim, for_loop_dim):
        for indent in range(1 + (eachdim - global_id_dim)):
            gufunc_txt += "    "

    return gufunc_txt


def _wrap_loop_body(loop_body):
    blocks = loop_body.copy()  # shallow copy is enough
    first_label = min(blocks.keys())
    last_label = max(blocks.keys())
    loc = blocks[last_label].loc
    blocks[last_label].body.append(ir.Jump(first_label, loc))
    return blocks


def _unwrap_loop_body(loop_body):
    last_label = max(loop_body.keys())
    loop_body[last_label].body = loop_body[last_label].body[:-1]


def _find_setitems_block(setitems, block, typemap):
    for inst in block.body:
        if isinstance(inst, ir.StaticSetItem) or isinstance(inst, ir.SetItem):
            setitems.add(inst.target.name)
        elif isinstance(inst, parfor.Parfor):
            _find_setitems_block(setitems, inst.init_block, typemap)
            _find_setitems_body(setitems, inst.loop_body, typemap)


def _find_setitems_body(setitems, loop_body, typemap):
    """
    Find the arrays that are written into (goes into setitems)
    """
    for label, block in loop_body.items():
        _find_setitems_block(setitems, block, typemap)


def create_kernel_for_parfor(
    lowerer,
    parfor_node,
    typemap,
    flags,
    loop_ranges,
    has_aliases,
    races,
    parfor_outputs,
):
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

    for start, stop, _ in loop_ranges:
        if isinstance(start, ir.Var):
            parfor_params.add(start.name)
        if isinstance(stop, ir.Var):
            parfor_params.add(stop.name)

    # Get all parfor reduction vars, and operators.
    typemap = lowerer.fndesc.typemap

    parfor_redvars, parfor_reddict = parfor.get_parfor_reductions(
        lowerer.func_ir, parfor_node, parfor_params, lowerer.fndesc.calltypes
    )
    has_reduction = False if len(parfor_redvars) == 0 else True

    # if has_reduction:
    #     raise NotImplementedError

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

    # param_types_addrspaces = copy.copy(param_types)

    # Calculate types of args passed to gufunc.
    func_arg_types = [typemap[v] for v in (parfor_inputs + parfor_outputs)]

    # addrspaces = _get_param_addrspace(parfor_params, typemap)

    # if len(param_types_addrspaces) != len(addrspaces):
    #     raise AssertionError

    # FIXME:: Most probably we need to convert DpnpNdArray type to UsmNdArray
    # for i in range(len(param_types_addrspaces)):
    #     if addrspaces[i] is not None:
    #         # Convert numba.types.Array to numba_dpex.core.types.Array data
    #         # type. Our Array type allows us to specify an address space for the
    #         # data and other pointer arguments for the array.
    #         param_types_addrspaces[i] = npytypes_array_to_dpex_array(
    #             param_types_addrspaces[i], addrspaces[i]
    #         )

    # if config.DEBUG_ARRAY_OPT >= 1:
    #     for a in param_types:
    #         print(a, type(a))
    #         if isinstance(a, types.npytypes.Array):
    #             print("addrspace:", a.addrspace)
    #     print("func_arg_types = ", func_arg_types, type(func_arg_types))

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

    # Determine the unique names of the scheduling and gufunc functions.
    gufunc_name = "__numba_parfor_gufunc_%s" % (parfor_node.id)

    gufunc_txt = ""

    # Create the gufunc function.
    gufunc_txt += "def " + gufunc_name
    gufunc_txt += "(" + (", ".join(parfor_params)) + "):\n"

    gufunc_txt += _generate_kernel_indexing(
        parfor_dim, legal_loop_indices, loop_ranges, param_dict, has_reduction, parfor_redvars, typemap
    )

    # Add the sentinel assignment so that we can find the loop body position
    # in the IR.
    gufunc_txt += "    "
    gufunc_txt += sentinel_name + " = 0\n"

    if has_reduction:
        # Generate local_sum[local_id0] = redvar, for each reduction variable
        for redvar in redvars:
            gufunc_txt += (
                "    "
                + f"local_sums_{redvar}[local_id0] = {redvar}\n"
            )

        gufunc_txt += (
            "    stride0 = group_size0 // 2\n" +
            "    while stride0 > 0:\n" +
            "        dpex.barrier(dpex.LOCAL_MEM_FENCE)\n" +
            "        if local_id0 < stride0:\n"
        )

        for redvar in redvars:
            gufunc_txt += (
                "            "
                + f"local_sums_{redvar}[local_id0] += local_sums_{redvar}[local_id0 + stride0]\n"
            )

        gufunc_txt += (
            "        stride0 >>= 1\n"
        )

        gufunc_txt += (
            "    if local_id0 == 0:\n"
        )
        for redvar in redvars:
            gufunc_txt += (
                "            "
                + f"partial_sums_{redvar}[group_id0] = local_sums_{redvar}[0]\n"
            )

    # gufunc returns nothing
    gufunc_txt += "    return None\n"

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_txt = ", type(gufunc_txt), "\n", gufunc_txt)
        sys.stdout.flush()
    # Force gufunc outline into existence.
    globls = {"dpnp": dpnp, "numba": numba, "dpex": dpex}
    locls = {}
    exec(gufunc_txt, globls, locls)
    gufunc_func = locls[gufunc_name]

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_func = ", type(gufunc_func), "\n", gufunc_func)
    # Get the IR for the gufunc outline.
    gufunc_ir = compiler.run_frontend(gufunc_func)

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump ", type(gufunc_ir))
        gufunc_ir.dump()
        print("loop_body dump ", type(loop_body))
        _print_body(loop_body)

    # rename all variables in gufunc_ir afresh
    var_table = get_name_var_table(gufunc_ir.blocks)
    new_var_dict = {}
    reserved_names = (
        [sentinel_name] + list(param_dict.values()) + legal_loop_indices
    )
    for name, var in var_table.items():
        if not (name in reserved_names):
            new_var_dict[name] = mk_unique_var(name)
    replace_var_names(gufunc_ir.blocks, new_var_dict)
    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir dump after renaming ")
        gufunc_ir.dump()

    gufunc_param_types = param_types

    if config.DEBUG_ARRAY_OPT:
        print(
            "gufunc_param_types = ",
            type(gufunc_param_types),
            "\n",
            gufunc_param_types,
        )

    gufunc_stub_last_label = max(gufunc_ir.blocks.keys()) + 1

    # Add gufunc stub last label to each parfor.loop_body label to prevent
    # label conflicts.
    loop_body = add_offset_to_labels(loop_body, gufunc_stub_last_label)
    # new label for splitting sentinel block
    new_label = max(loop_body.keys()) + 1

    # If enabled, add a print statement after every assignment.
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        _dbgprint_after_each_array_assignments(lowerer, loop_body, typemap)

    if config.DEBUG_ARRAY_OPT:
        print("parfor loop body")
        _print_body(loop_body)

    # ----------- Remove as CPU LICM does nto work for kernels ---------------##

    # wrapped_blocks = _wrap_loop_body(loop_body)
    # # hoisted, not_hoisted = hoist(parfor_params, loop_body,
    # #                             typemap, wrapped_blocks)
    setitems = set()
    _find_setitems_body(setitems, loop_body, typemap)

    # hoisted = []
    # not_hoisted = []

    # start_block = gufunc_ir.blocks[min(gufunc_ir.blocks.keys())]
    # start_block.body = start_block.body[:-1] + hoisted + [start_block.body[-1]]
    # _unwrap_loop_body(loop_body)

    # # store hoisted into diagnostics
    # diagnostics = lowerer.metadata["parfor_diagnostics"]
    # diagnostics.hoist_info[parfor_node.id] = {
    #     "hoisted": hoisted,
    #     "not_hoisted": not_hoisted,
    # }

    # lowerer.metadata["parfor_diagnostics"].extra_info[
    #     str(parfor_node.id)
    # ] = str(dpctl.get_current_queue().get_sycl_device().name)

    # ----------- Remove as CPU LICM does nto work for kernels ---------------##

    # Search all the block in the gufunc outline for the sentinel assignment.
    for label, block in gufunc_ir.blocks.items():
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
                body_first_label = min(loop_body.keys())

                # The previous block jumps to the minimum labelled block of the
                # parfor body.
                prev_block.append(ir.Jump(body_first_label, loc))
                # Add all the parfor loop body blocks to the gufunc function's
                # IR.
                for loop, b in loop_body.items():
                    gufunc_ir.blocks[loop] = b
                body_last_label = max(loop_body.keys())
                gufunc_ir.blocks[new_label] = block
                gufunc_ir.blocks[label] = prev_block
                # Add a jump from the last parfor body block to the block
                # containing statements after the sentinel.
                gufunc_ir.blocks[body_last_label].append(
                    ir.Jump(new_label, loc)
                )
                break
        else:
            continue
        break

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir last dump before renaming")
        gufunc_ir.dump()

    gufunc_ir.blocks = rename_labels(gufunc_ir.blocks)
    remove_dels(gufunc_ir.blocks)

    old_alias = flags.noalias
    if not has_aliases:
        if config.DEBUG_ARRAY_OPT:
            print("No aliases found so adding noalias flag.")
        flags.noalias = True

    remove_dead(gufunc_ir.blocks, gufunc_ir.arg_names, gufunc_ir, typemap)

    if config.DEBUG_ARRAY_OPT:
        print("gufunc_ir after remove dead")
        gufunc_ir.dump()

    kernel_sig = signature(types.none, *gufunc_param_types)

    if config.DEBUG_ARRAY_OPT:
        sys.stdout.flush()

    # if config.DEBUG_ARRAY_OPT:
    #     print("before DUFunc inlining".center(80, "-"))
    #     gufunc_ir.dump()

    # # Inlining all DUFuncs
    # dufunc_inliner(
    #     gufunc_ir,
    #     lowerer.fndesc.calltypes,
    #     typemap,
    #     lowerer.context.typing_context,
    #     lowerer.context,
    # )

    if config.DEBUG_ARRAY_OPT:
        print("after DUFunc inline".center(80, "-"))
        gufunc_ir.dump()

    exec_queue = determine_kernel_launch_queue(
        args=parfor_args, argtypes=gufunc_param_types, kernel_name=gufunc_name
    )

    sycl_kernel = _compile_kernel_parfor(
        exec_queue,
        gufunc_name,
        gufunc_ir,
        gufunc_param_types,
        debug=flags.debuginfo,
    )

    flags.noalias = old_alias

    if config.DEBUG_ARRAY_OPT:
        print("kernel_sig = ", kernel_sig)

    return (
        sycl_kernel,
        parfor_args,
        kernel_sig,
        func_arg_types,
        setitems,
        exec_queue,
    )
