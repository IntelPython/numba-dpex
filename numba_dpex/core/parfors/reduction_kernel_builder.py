# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dpctl
from numba.core import types
from numba.core.errors import NumbaParallelSafetyWarning
from numba.core.ir_utils import (
    add_offset_to_labels,
    get_name_var_table,
    get_unused_var_name,
    legalize_names,
    mk_unique_var,
    remove_dels,
    rename_labels,
    replace_var_names,
)
from numba.core.typing import signature

from numba_dpex.core.parfors.reduction_helper import ReductionKernelVariables
from numba_dpex.core.types import DpctlSyclQueue
from numba_dpex.core.types.kernel_api.index_space_ids import NdItemType
from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType

from .kernel_builder import _print_body  # saved for debug
from .kernel_builder import (
    ParforKernel,
    _compile_kernel_parfor,
    _to_scalar_from_0d,
    update_sentinel,
)
from .kernel_templates.reduction_template import (
    RemainderReduceIntermediateKernelTemplate,
    TreeReduceIntermediateKernelTemplate,
)


def create_reduction_main_kernel_for_parfor(
    loop_ranges,
    parfor_node,
    typemap,
    flags,
    has_aliases,
    reductionKernelVar: ReductionKernelVariables,
    parfor_reddict=None,
):
    """
    Creates a numba_dpex.kernel function for reduction main kernel.
    """

    loc = parfor_node.init_block.loc
    parfor_dim = len(parfor_node.loop_nests)

    for race in parfor_node.races:
        msg = (
            "Variable %s used in parallel loop may be written "
            "to simultaneously by multiple workers and may result "
            "in non-deterministic or unintended results." % race
        )
        warnings.warn(NumbaParallelSafetyWarning(msg, loc))

    loop_body_var_table = get_name_var_table(reductionKernelVar.loop_body)
    sentinel_name = get_unused_var_name("__sentinel__", loop_body_var_table)

    # Determine the unique names of the scheduling and kernel functions.
    kernel_name = "__dpex_reduction_parfor_%s" % (parfor_node.id)

    # swap s.2 (redvar) with partial_sum
    for i, name in enumerate(reductionKernelVar.parfor_params):
        try:
            tmp = reductionKernelVar.parfor_redvars_to_redarrs[name][0]
            nameNew = reductionKernelVar.parfor_redarrs_legal_dict[tmp]
            reductionKernelVar.parfor_legalized_params[i] = nameNew
            reductionKernelVar.param_types[i] = _to_scalar_from_0d(typemap[tmp])
            reductionKernelVar.func_arg_types[i] = _to_scalar_from_0d(
                typemap[tmp]
            )

        except KeyError:
            pass

    parfor_params = reductionKernelVar.parfor_params.copy()
    parfor_legalized_params = reductionKernelVar.parfor_legalized_params.copy()
    parfor_param_types = reductionKernelVar.param_types.copy()
    local_accessors_dict = {}
    for k, v in reductionKernelVar.redvars_legal_dict.items():
        la_var = "local_sums_" + v
        local_accessors_dict[k] = la_var
        idx = reductionKernelVar.parfor_params.index(k)
        arr_ty = reductionKernelVar.param_types[idx]
        la_ty = LocalAccessorType(parfor_dim, arr_ty.dtype)

        parfor_params.append(la_var)
        parfor_legalized_params.append(la_var)
        parfor_param_types.append(la_ty)

    kernel_template = TreeReduceIntermediateKernelTemplate(
        kernel_name=kernel_name,
        kernel_params=parfor_legalized_params,
        ivar_names=reductionKernelVar.legal_loop_indices,
        sentinel_name=sentinel_name,
        loop_ranges=loop_ranges,
        param_dict=reductionKernelVar.param_dict,
        parfor_dim=parfor_dim,
        redvars=reductionKernelVar.parfor_redvars,
        parfor_args=parfor_params,
        parfor_reddict=parfor_reddict,
        redvars_dict=reductionKernelVar.redvars_legal_dict,
        local_accessors_dict=local_accessors_dict,
        typemap=typemap,
    )
    kernel_ir = kernel_template.kernel_ir

    for i, name in enumerate(reductionKernelVar.parfor_params):
        try:
            tmp = reductionKernelVar.parfor_redvars_to_redarrs[name][0]
            reductionKernelVar.parfor_params[i] = tmp
        except KeyError:
            pass

    # rename all variables in kernel_ir afresh
    var_table = get_name_var_table(kernel_ir.blocks)
    new_var_dict = {}
    reserved_names = (
        [sentinel_name]
        + list(reductionKernelVar.param_dict.values())
        + reductionKernelVar.legal_loop_indices
    )
    for name, _ in var_table.items():
        if not (name in reserved_names):
            new_var_dict[name] = mk_unique_var(name)

    replace_var_names(kernel_ir.blocks, new_var_dict)
    kernel_param_types = parfor_param_types
    kernel_stub_last_label = max(kernel_ir.blocks.keys()) + 1
    # Add kernel stub last label to each parfor.loop_body label to prevent
    # label conflicts.
    loop_body = add_offset_to_labels(
        reductionKernelVar.loop_body, kernel_stub_last_label
    )
    # new label for splitting sentinel block
    new_label = max(loop_body.keys()) + 1

    update_sentinel(kernel_ir, sentinel_name, loop_body, new_label)

    # FIXME: Why rename and remove dels causes the partial_sum array update
    # instructions to be removed.
    kernel_ir.blocks = rename_labels(kernel_ir.blocks)
    remove_dels(kernel_ir.blocks)

    old_alias = flags.noalias

    if not has_aliases:
        flags.noalias = True

    # The first argument to a range kernel is a kernel_api.NdItem object. The
    # ``NdItem`` object is used by the kernel_api.spirv backend to generate the
    # correct SPIR-V indexing instructions. Since, the argument is not something
    # available originally in the kernel_param_types, we add it at this point to
    # make sure the kernel signature matches the actual generated code.
    ty_item = NdItemType(parfor_dim)
    kernel_param_types = (ty_item, *kernel_param_types)
    kernel_sig = signature(types.none, *kernel_param_types)

    # FIXME: A better design is required so that we do not have to create a
    # queue every time.
    ty_queue: DpctlSyclQueue = typemap[
        reductionKernelVar.parfor_params[0]
    ].queue
    exec_queue = dpctl.get_device_cached_queue(ty_queue.sycl_device)

    sycl_kernel = _compile_kernel_parfor(
        exec_queue,
        kernel_name,
        kernel_ir,
        kernel_param_types,
        debug=flags.debuginfo,
    )

    flags.noalias = old_alias

    parfor_params = (
        reductionKernelVar.parfor_params.copy()
        + parfor_params[len(reductionKernelVar.parfor_params) :]  # noqa: $203
    )

    return ParforKernel(
        name=kernel_name,
        kernel=sycl_kernel,
        signature=kernel_sig,
        kernel_args=parfor_params,
        kernel_arg_types=parfor_param_types,
        queue=exec_queue,
        local_accessors=set(local_accessors_dict.values()),
        work_group_size=reductionKernelVar.work_group_size,
    )


def create_reduction_remainder_kernel_for_parfor(
    parfor_node,
    typemap,
    flags,
    has_aliases,
    reductionKernelVar,
    parfor_reddict,
    reductionHelperList,
):
    """
    Creates a numba_dpex.kernel function for a reduction remainder kernel.
    """

    loc = parfor_node.init_block.loc

    for race in parfor_node.races:
        msg = (
            "Variable %s used in parallel loop may be written "
            "to simultaneously by multiple workers and may result "
            "in non-deterministic or unintended results." % race
        )
        warnings.warn(NumbaParallelSafetyWarning(msg, loc))

    loop_body_var_table = get_name_var_table(reductionKernelVar.loop_body)
    sentinel_name = get_unused_var_name("__sentinel__", loop_body_var_table)

    global_size_var_name = []
    global_size_mod_var_name = []
    partial_sum_var_name = []
    partial_sum_size_var_name = []
    final_sum_var_name = []

    for i, _ in enumerate(reductionKernelVar.parfor_redvars):
        reductionHelper = reductionHelperList[i]
        name = reductionHelper.partial_sum_var.name
        partial_sum_var_name.append(name)

        name = reductionHelper.global_size_var.name
        global_size_var_name.append(name)

        name = reductionHelper.global_size_mod_var.name
        global_size_mod_var_name.append(name)

        name = reductionHelper.partial_sum_size_var.name
        partial_sum_size_var_name.append(name)

        name = reductionHelper.final_sum_var.name
        final_sum_var_name.append(name)

    kernel_name = "__dpex_redection_parfor_%s_remainder" % (parfor_node.id)

    partial_sum_var_dict = legalize_names(partial_sum_var_name)
    global_size_var_dict = legalize_names(global_size_var_name)
    global_size_mod_var_dict = legalize_names(global_size_mod_var_name)
    partial_sum_size_var_dict = legalize_names(partial_sum_size_var_name)
    final_sum_var_dict = legalize_names(final_sum_var_name)

    partial_sum_var_legal_name = [
        partial_sum_var_dict[v] for v in partial_sum_var_dict
    ]

    global_size_var_legal_name = [
        global_size_var_dict[v] for v in global_size_var_dict
    ]
    global_size_mod_var_legal_name = [
        global_size_mod_var_dict[v] for v in global_size_mod_var_dict
    ]
    partial_sum_size_var_legal_name = [
        partial_sum_size_var_dict[v] for v in partial_sum_size_var_dict
    ]
    final_sum_var_legal_name = [
        final_sum_var_dict[v] for v in final_sum_var_dict
    ]

    kernel_template = RemainderReduceIntermediateKernelTemplate(
        kernel_name=kernel_name,
        kernel_params=reductionKernelVar.parfor_legalized_params,
        sentinel_name=sentinel_name,
        legal_loop_indices=reductionKernelVar.legal_loop_indices,
        redvars_dict=reductionKernelVar.redvars_legal_dict,
        redvars=reductionKernelVar.parfor_redvars,
        parfor_reddict=parfor_reddict,
        typemap=typemap,
        global_size_var_name=global_size_var_legal_name,
        global_size_mod_var_name=global_size_mod_var_legal_name,
        partial_sum_size_var_name=partial_sum_size_var_legal_name,
        partial_sum_var_name=partial_sum_var_legal_name,
        final_sum_var_name=final_sum_var_legal_name,
        reductionKernelVar=reductionKernelVar,
    )
    kernel_ir = kernel_template.kernel_ir

    var_table = get_name_var_table(kernel_ir.blocks)
    new_var_dict = {}
    reserved_names = (
        [sentinel_name]
        + list(reductionKernelVar.param_dict.values())
        + reductionKernelVar.legal_loop_indices
    )
    for name, _ in var_table.items():
        if not (name in reserved_names):
            new_var_dict[name] = mk_unique_var(name)
    replace_var_names(kernel_ir.blocks, new_var_dict)

    for i, _ in enumerate(reductionKernelVar.parfor_redvars):
        if reductionHelperList[i].global_size_var is not None:
            reductionKernelVar.parfor_params.append(global_size_var_name[i])
            reductionKernelVar.parfor_legalized_params.append(
                global_size_var_legal_name[i]
            )
            reductionKernelVar.param_types.append(
                _to_scalar_from_0d(typemap[global_size_var_name[i]])
            )
            reductionKernelVar.func_arg_types.append(
                _to_scalar_from_0d(typemap[global_size_var_name[i]])
            )

        if reductionHelperList[i].global_size_mod_var is not None:
            reductionKernelVar.parfor_params.append(global_size_mod_var_name[i])
            reductionKernelVar.parfor_legalized_params.append(
                global_size_mod_var_legal_name[i]
            )

            reductionKernelVar.param_types.append(
                _to_scalar_from_0d(typemap[global_size_mod_var_name[i]])
            )
            reductionKernelVar.func_arg_types.append(
                _to_scalar_from_0d(typemap[global_size_mod_var_name[i]])
            )

        if reductionHelperList[i].partial_sum_size_var is not None:
            reductionKernelVar.parfor_params.append(
                partial_sum_size_var_name[i]
            )
            reductionKernelVar.parfor_legalized_params.append(
                partial_sum_size_var_legal_name[i]
            )
            reductionKernelVar.param_types.append(
                _to_scalar_from_0d(typemap[partial_sum_size_var_name[i]])
            )
            reductionKernelVar.func_arg_types.append(
                _to_scalar_from_0d(typemap[partial_sum_size_var_name[i]])
            )
        if reductionHelperList[i].final_sum_var is not None:
            reductionKernelVar.parfor_params.append(final_sum_var_name[i])
            reductionKernelVar.parfor_legalized_params.append(
                final_sum_var_legal_name[i]
            )
            reductionKernelVar.param_types.append(
                _to_scalar_from_0d(typemap[final_sum_var_name[i]])
            )
            reductionKernelVar.func_arg_types.append(
                _to_scalar_from_0d(typemap[final_sum_var_name[i]])
            )

    kernel_param_types = reductionKernelVar.param_types

    kernel_stub_last_label = max(kernel_ir.blocks.keys()) + 1

    # Add kernel stub last label to each parfor.loop_body label to prevent
    # label conflicts.
    loop_body = add_offset_to_labels(
        reductionKernelVar.loop_body, kernel_stub_last_label
    )
    # new label for splitting sentinel block
    new_label = max(loop_body.keys()) + 1

    update_sentinel(kernel_ir, sentinel_name, loop_body, new_label)

    old_alias = flags.noalias
    if not has_aliases:
        flags.noalias = True

    kernel_sig = signature(types.none, *kernel_param_types)

    # FIXME: A better design is required so that we do not have to create a
    # queue every time.
    ty_queue: DpctlSyclQueue = typemap[
        reductionKernelVar.parfor_params[0]
    ].queue
    exec_queue = dpctl.get_device_cached_queue(ty_queue.sycl_device)

    sycl_kernel = _compile_kernel_parfor(
        exec_queue,
        kernel_name,
        kernel_ir,
        kernel_param_types,
        debug=flags.debuginfo,
    )

    flags.noalias = old_alias

    return ParforKernel(
        name=kernel_name,
        kernel=sycl_kernel,
        signature=kernel_sig,
        kernel_args=reductionKernelVar.parfor_params,
        kernel_arg_types=reductionKernelVar.func_arg_types,
        queue=exec_queue,
    )
