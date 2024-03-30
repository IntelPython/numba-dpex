# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Various utility functions and classes to aid LLVM IR building.

"""

from numba_dpex.utils.llvm_codegen_helpers import (
    LLVMTypes,
    create_null_ptr,
    get_llvm_ptr_type,
    get_llvm_type,
    get_one,
    get_zero,
)

__all__ = [
    "LLVMTypes",
    "get_llvm_type",
    "get_llvm_ptr_type",
    "create_null_ptr",
    "get_zero",
    "get_one",
]
