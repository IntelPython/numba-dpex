# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from llvmlite import ir as llvmir

import numba_dpex as dpex
from numba_dpex import DpctlSyclQueue, DpnpNdArray, int64
from numba_dpex.core.types.kernel_api.index_space_ids import NdItemType
from numba_dpex.core.types.kernel_api.local_accessor import LocalAccessorType
from numba_dpex.kernel_api import (
    AddressSpace,
    MemoryScope,
    NdItem,
    group_barrier,
)


def kernel_func(nd_item: NdItem, a, slm):
    i = nd_item.get_global_linear_id()
    j = nd_item.get_local_linear_id()

    slm[j] = 100
    group_barrier(nd_item.get_group(), MemoryScope.WORK_GROUP)

    a[i] += slm[j]


def test_codegen_local_accessor_kernel_arg():
    """Tests if a kernel with a local accessor argument is generated with
    expected local address space pointer argument.
    """

    queue_ty = DpctlSyclQueue(dpctl.SyclQueue())
    i64arr_ty = DpnpNdArray(ndim=1, dtype=int64, layout="C", queue=queue_ty)
    slm_ty = LocalAccessorType(ndim=1, dtype=int64)
    disp = dpex.kernel(inline_threshold=3)(kernel_func)
    dmm = disp.targetctx.data_model_manager

    i64arr_ty_flattened_arg_count = dmm.lookup(i64arr_ty).flattened_field_count
    slm_ty_model = dmm.lookup(slm_ty)
    slm_ty_flattened_arg_count = slm_ty_model.flattened_field_count
    slm_ptr_pos = slm_ty_model.get_field_position("data")

    llargtys = disp.targetctx.get_arg_packer([i64arr_ty, slm_ty]).argument_types

    # Go over all the arguments to the spir_kernel_func and assert two things:
    # a) Number of arguments == i64arr_ty_flattened_arg_count
    #    + slm_ty_flattened_arg_count
    # b) The argument corresponding to the data attribute of the local accessor
    # argument is a pointer in address space local address space

    num_kernel_args = 0
    slm_data_ptr_arg = None
    for kernel_arg in llargtys:
        if num_kernel_args == i64arr_ty_flattened_arg_count + slm_ptr_pos:
            slm_data_ptr_arg = kernel_arg
        num_kernel_args += 1
    assert (
        num_kernel_args
        == i64arr_ty_flattened_arg_count + slm_ty_flattened_arg_count
    )
    assert isinstance(slm_data_ptr_arg, llvmir.PointerType)
    assert slm_data_ptr_arg.addrspace == AddressSpace.LOCAL
