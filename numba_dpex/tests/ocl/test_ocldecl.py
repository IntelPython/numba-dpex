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

import pytest
from numba import types

import numba_dpex as dpex
from numba_dpex.ocl import ocldecl
from numba_dpex.ocl.ocldecl import registry


@pytest.mark.parametrize(
    "fname",
    [
        "get_global_id",
        "get_local_id",
        "get_group_id",
        "get_num_groups",
        "get_work_dim",
        "get_global_size",
        "get_local_size",
        "barrier",
        "mem_fence",
        "sub_group_barrier",
    ],
)
def test_registry(fname):
    function = getattr(dpex, fname)
    template = getattr(ocldecl, f"Ocl_{fname}")

    assert template in registry.functions
    assert (function, types.Function(template)) in registry.globals
