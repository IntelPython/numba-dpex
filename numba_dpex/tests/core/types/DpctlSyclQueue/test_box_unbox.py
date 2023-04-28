# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing and unboxing of types supported inside dpjit
"""

import dpctl

from numba_dpex import dpjit


def test_boxing_unboxing():
    """Tests basic boxing and unboxing of a dpctl.SyclQueue object.

    Checks if we can pass in and return a dpctl.SyclQueue object to and
    from a dpjit decorated function.
    """

    @dpjit
    def func(a):
        return a

    q = dpctl.SyclQueue()
    o = func(q)
    assert id(o) == id(q)
