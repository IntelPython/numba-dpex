#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for common tools"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .gdb import breakpoint_by_function, breakpoint_by_mark, setup_breakpoint

pytestmark = skip_no_gdb


@pytest.mark.parametrize(
    "file_name, mark, expected",
    [("side-by-side.py", "Set breakpoint here", "side-by-side.py:16")],
)
def test_breakpoint_by_mark(file_name, mark, expected):
    assert expected == breakpoint_by_mark(file_name, mark)


@pytest.mark.parametrize(
    "file_name, function, expected",
    [("side-by-side.py", "common_loop_body", "side-by-side.py:16")],
)
def test_breakpoint_by_function(file_name, function, expected):
    assert expected == breakpoint_by_function(file_name, function)


@pytest.mark.parametrize(
    "breakpoint, script, expected_location, expected_line",
    [
        (
            "simple_sum.py:14",
            "simple_sum.py",
            "simple_sum.py:14",
            r"14\s+c\[i\] = a\[i\] \+ b\[i\]",
        ),
        ("simple_sum.py:14", "simple_sum.py", "simple_sum.py:14", None),
        ("simple_sum.py:14", "simple_sum.py", None, None),
        ("simple_sum.py:14", None, None, None),
        ("simple_sum.py:data_parallel_sum", None, None, None),
        ("data_parallel_sum", "simple_sum.py", None, None),
    ],
)
def test_setup_breakpoint(
    app, breakpoint, script, expected_location, expected_line
):
    if (
        breakpoint == "simple_sum.py:data_parallel_sum"
        or breakpoint == "data_parallel_sum"
    ):
        pytest.xfail(
            "Expected failures for these files."
        )  # TODO: https://github.com/IntelPython/numba-dpex/issues/1242

    setup_breakpoint(app, breakpoint, script, expected_location, expected_line)
