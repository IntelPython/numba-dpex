# Copyright 2021 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

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
