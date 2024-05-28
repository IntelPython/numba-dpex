# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

from numba.core import types
from numba.core.errors import NumbaParallelSafetyWarning
from numba.core.ir_utils import (
    get_name_var_table,
    get_unused_var_name,
    legalize_names,
)
from numba.core.typing import signature

from numba_dpex.core.decorators import kernel
from numba_dpex.core.parfors.parfor_sentinel_replace_pass import (
    ParforBodyArguments,
)
from numba_dpex.core.parfors.reduction_helper import ReductionKernelVariables
from numba_dpex.core.types.kernel_api.index_space_ids import NdItemType
from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType
from numba_dpex.core.utils.call_kernel_builder import SPIRVKernelModule
from numba_dpex.kernel_api_impl.spirv.dispatcher import (
    SPIRVKernelDispatcher,
    _SPIRVKernelCompileResult,
)

from .kernel_builder import ParforKernel, _to_scalar_from_0d
from .kernel_templates.reduction_template import (
    RemainderReduceIntermediateKernelTemplate,
    TreeReduceIntermediateKernelTemplate,
)


def create_reduction_main_kernel_for_parfor(
    loop_ranges,
    parfor_node,
    typemap,
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

    for i, name in enumerate(reductionKernelVar.parfor_params):
        try:
            tmp = reductionKernelVar.parfor_redvars_to_redarrs[name][0]
            reductionKernelVar.parfor_params[i] = tmp
        except KeyError:
            pass

    kernel_dispatcher: SPIRVKernelDispatcher = kernel(
        kernel_template.py_func,
        _parfor_body_args=ParforBodyArguments(
            loop_body=reductionKernelVar.loop_body,
            param_dict=reductionKernelVar.param_dict,
            legal_loop_indices=reductionKernelVar.legal_loop_indices,
        ),
    )

    # The first argument to a range kernel is a kernel_api.NdItem object. The
    # ``NdItem`` object is used by the kernel_api.spirv backend to generate the
    # correct SPIR-V indexing instructions. Since, the argument is not something
    # available originally in the kernel_param_types, we add it at this point to
    # make sure the kernel signature matches the actual generated code.
    ty_item = NdItemType(parfor_dim)
    kernel_param_types = (ty_item, *parfor_param_types)
    kernel_sig = signature(types.none, *kernel_param_types)

    kcres: _SPIRVKernelCompileResult = kernel_dispatcher.get_compile_result(
        types.void(*kernel_param_types)  # kernel signature
    )
    kernel_module: SPIRVKernelModule = kcres.kernel_device_ir_module

    parfor_params = (
        reductionKernelVar.parfor_params.copy()
        + parfor_params[len(reductionKernelVar.parfor_params) :]  # noqa: $203
    )

    return ParforKernel(
        signature=kernel_sig,
        kernel_args=parfor_params,
        kernel_arg_types=parfor_param_types,
        local_accessors=set(local_accessors_dict.values()),
        work_group_size=reductionKernelVar.work_group_size,
        kernel_module=kernel_module,
    )


def create_reduction_remainder_kernel_for_parfor(
    parfor_node,
    typemap,
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

    kernel_dispatcher: SPIRVKernelDispatcher = kernel(
        kernel_template.py_func,
        _parfor_body_args=ParforBodyArguments(
            loop_body=reductionKernelVar.loop_body,
            param_dict=reductionKernelVar.param_dict,
            legal_loop_indices=reductionKernelVar.legal_loop_indices,
        ),
    )

    kernel_param_types = reductionKernelVar.param_types

    kernel_sig = signature(types.none, *kernel_param_types)

    kcres: _SPIRVKernelCompileResult = kernel_dispatcher.get_compile_result(
        types.void(*kernel_param_types)  # kernel signature
    )
    kernel_module: SPIRVKernelModule = kcres.kernel_device_ir_module

    return ParforKernel(
        signature=kernel_sig,
        kernel_args=reductionKernelVar.parfor_params,
        kernel_arg_types=reductionKernelVar.func_arg_types,
        kernel_module=kernel_module,
    )
