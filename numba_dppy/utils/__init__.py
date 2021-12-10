# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Various utility functions and classes to aid LLVM IR building.

"""
from numba_dppy.utils.array_utils import (
    as_usm_obj,
    copy_from_numpy_to_usm_obj,
    copy_to_numpy_from_usm_obj,
    get_info_from_suai,
    has_usm_memory,
)
from numba_dppy.utils.constants import address_space, calling_conv
from numba_dppy.utils.llvm_codegen_helpers import (
    LLVMTypes,
    create_null_ptr,
    get_llvm_ptr_type,
    get_llvm_type,
    get_one,
    get_zero,
)
from numba_dppy.utils.messages import (
    IndeterminateExecutionQueueError_msg,
    cfd_ctx_mgr_wrng_msg,
    mix_datatype_err_msg,
)
from numba_dppy.utils.misc import (
    IndeterminateExecutionQueueError,
    assert_no_return,
)
from numba_dppy.utils.type_conversion_fns import (
    npytypes_array_to_dppy_array,
    suai_to_dppy_array_type,
)

__all__ = [
    "LLVMTypes",
    "get_llvm_type",
    "get_llvm_ptr_type",
    "create_null_ptr",
    "get_zero",
    "get_one",
    "npytypes_array_to_dppy_array",
    "npytypes_array_to_dppy_array",
    "suai_to_dppy_array_type",
    "address_space",
    "calling_conv",
    "has_usm_memory",
    "as_usm_obj",
    "copy_from_numpy_to_usm_obj",
    "copy_to_numpy_from_usm_obj",
    "assert_no_return",
    "IndeterminateExecutionQueueError",
    "cfd_ctx_mgr_wrng_msg",
    "IndeterminateExecutionQueueError_msg",
    "mix_datatype_err_msg",
    "get_info_from_suai",
]
