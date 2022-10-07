# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Various utility functions and classes to aid LLVM IR building.

"""
from numba_dpex.utils.array_utils import (
    as_usm_obj,
    copy_from_numpy_to_usm_obj,
    copy_to_numpy_from_usm_obj,
    get_info_from_suai,
    has_usm_memory,
)
from numba_dpex.utils.constants import address_space, calling_conv
from numba_dpex.utils.llvm_codegen_helpers import (
    LLVMTypes,
    create_null_ptr,
    get_llvm_ptr_type,
    get_llvm_type,
    get_one,
    get_zero,
)
from numba_dpex.utils.messages import (
    IndeterminateExecutionQueueError_msg,
    cfd_ctx_mgr_wrng_msg,
    mix_datatype_err_msg,
)
from numba_dpex.utils.misc import IndeterminateExecutionQueueError
from numba_dpex.utils.type_conversion_fns import (
    npytypes_array_to_dpex_array,
    suai_to_dpex_array,
)

__all__ = [
    "LLVMTypes",
    "get_llvm_type",
    "get_llvm_ptr_type",
    "create_null_ptr",
    "get_zero",
    "get_one",
    "npytypes_array_to_dpex_array",
    "npytypes_array_to_dpex_array",
    "suai_to_dpex_array",
    "address_space",
    "calling_conv",
    "has_usm_memory",
    "as_usm_obj",
    "copy_from_numpy_to_usm_obj",
    "copy_to_numpy_from_usm_obj",
    "IndeterminateExecutionQueueError",
    "cfd_ctx_mgr_wrng_msg",
    "IndeterminateExecutionQueueError_msg",
    "mix_datatype_err_msg",
    "get_info_from_suai",
]
