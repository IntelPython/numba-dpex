# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for numba.core.runtime._nrt_python"""


def test_c_helpers():
    from numba.core.runtime._nrt_python import c_helpers

    functions = [
        "MemInfo_release",
    ]

    for fn_name in functions:
        assert fn_name in c_helpers
        assert isinstance(c_helpers[fn_name], int)
