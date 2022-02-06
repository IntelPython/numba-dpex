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


# Register the helper function in dppl_rt so that we can insert calls to them via llvmlite.
# for (
#     py_name,
#     c_address,
# ) in numba_dppy._usm_shared_allocator_ext.c_helpers.items():
#     llb.add_symbol(py_name, c_address)

import llvmlite.binding as llb

from numba_dppy import runtime


def test_llvm_symbol_registered():
    assert (
        llb.address_of_symbol("DPPY_RT_sycl_usm_array_from_python")
        == runtime._rt_python.DPPY_RT_sycl_usm_array_from_python
    )
