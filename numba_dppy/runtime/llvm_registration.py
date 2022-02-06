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
"""Registering symbols in llvm for calling it from generated code."""

import llvmlite.binding as llb

from numba_dppy.runtime import _rt_python

llb.add_symbol(
    "DPPY_RT_sycl_usm_array_from_python",
    _rt_python.DPPY_RT_sycl_usm_array_from_python,
)

llb.add_symbol(
    "DPPY_RT_sycl_usm_array_to_python_acqref",
    _rt_python.DPPY_RT_sycl_usm_array_to_python_acqref,
)
