# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Various utility functions and classes to aid LLVM IR building.

"""
from numba_dpex.utils.array_utils import (
    as_usm_obj,
    copy_from_numpy_to_usm_obj,
    copy_to_numpy_from_usm_obj,
    has_usm_memory,
)
from numba_dpex.utils.constants import address_space, calling_conv
from numba_dpex.utils.llvm_codegen_helpers import (
    LLVMTypes,
    get_llvm_ptr_type,
    get_llvm_type,
    get_nullptr,
    get_one,
    get_zero,
)

__all__ = [
    "LLVMTypes",
    "get_llvm_type",
    "get_llvm_ptr_type",
    "get_nullptr",
    "get_zero",
    "get_one",
    "address_space",
    "calling_conv",
    "has_usm_memory",
    "as_usm_obj",
    "copy_from_numpy_to_usm_obj",
    "copy_to_numpy_from_usm_obj",
]
