# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp

import numba_dpex as dpex
from numba_dpex import Range
from numba_dpex.kernel_api.flag_enum import FlagEnum


class MockFlags(FlagEnum):
    FLAG1 = 100
    FLAG2 = 200


@dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def update_with_flag(a):
    a[0] = MockFlags.FLAG1
    a[1] = MockFlags.FLAG2


def test_compilation_of_flag_enum():
    """Tests if a FlagEnum subclass can be used inside a kernel function."""
    a = dpnp.ones(10, dtype=dpnp.int64)
    dpex.call_kernel(update_with_flag, Range(10), a)

    assert a[0] == MockFlags.FLAG1
    assert a[1] == MockFlags.FLAG2
    for idx in range(2, 9):
        assert a[idx] == 1
