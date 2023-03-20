# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing and unboxing of types supported inside dpjit
"""

import dpctl
import pytest

from numba_dpex import dpjit


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(dpctl.SyclQueue()),
    ],
)
def test_boxing_unboxing(obj):
    @dpjit
    def func(a):
        return a

    o = func(obj)
    assert id(o) == id(obj)
