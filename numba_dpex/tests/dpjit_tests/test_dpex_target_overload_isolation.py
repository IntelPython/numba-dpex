# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests if dpex target overloads are not available at numba.njit and only
available at numba_dpex.dpjit.
"""

import pytest
from numba import njit
from numba.core import errors
from numba.extending import overload

from numba_dpex import dpjit
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME


def foo():
    return 1


@overload(foo, target=DPEX_TARGET_NAME)
def ol_foo():
    return lambda: 1


def bar():
    return foo()


def test_dpex_overload_from_njit():
    bar_njit = njit(bar)

    with pytest.raises(errors.TypingError):
        bar_njit()


def test_dpex_overload_from_dpjit():
    bar_dpjit = dpjit(bar)
    bar_dpjit()
