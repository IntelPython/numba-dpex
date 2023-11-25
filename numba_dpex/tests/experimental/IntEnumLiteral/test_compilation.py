# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba.core import types
from numba.extending import intrinsic, overload

import numba_dpex.experimental as exp_dpex
from numba_dpex import Range, dpjit
from numba_dpex.experimental.flag_enum import FlagEnum


class MockFlags(FlagEnum):
    FLAG1 = 100
    FLAG2 = 200


@exp_dpex.kernel(
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
    exp_dpex.call_kernel(update_with_flag, Range(10), a)

    assert a[0] == MockFlags.FLAG1
    assert a[1] == MockFlags.FLAG2
    for idx in range(2, 9):
        assert a[idx] == 1
