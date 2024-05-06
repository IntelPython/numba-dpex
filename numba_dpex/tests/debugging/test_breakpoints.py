#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Setting Breakpoints

https://www.sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html
"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .gdb import gdb

pytestmark = skip_no_gdb


@pytest.mark.parametrize(
    "breakpoint",
    [
        "side-by-side.py:15",
        "common_loop_body",
        "side-by-side.py:common_loop_body",
    ],
)
@pytest.mark.parametrize(
    "api",
    [
        "numba",
        "numba-ndpx-kernel",
    ],
)
@pytest.mark.parametrize(
    "condition, exp_var, exp_val",
    [
        ("param_a == 3", "param_a", 3),
        ("param_a == 7", "param_a", 7),
        (None, "param_a", r"[0-9]+"),  # No condition
    ],
)
def test_device_func_breakpoint(
    app: gdb, breakpoint, api, condition, exp_var, exp_val
):
    """Function breakpoints and argument initializing

    Test that it is possible to set conditional breakpoint at the beginning
    of the function and use a function argument in the condition.

    It is important that breakpoint by function name hits at the first line in
    the function body and not at the function definition line.

    Test for https://github.com/numba/numba/issues/7415
    SAT-4449
    """

    app.breakpoint(breakpoint, condition=condition)
    app.run(f"side-by-side.py --api={api}")
    app.expect_hit_breakpoint(expected_location="side-by-side.py:15")
    if exp_var is not None:
        app.print(exp_var, expected=exp_val)


@pytest.mark.parametrize(
    "condition, exp_var, exp_val",
    [
        ("i == 3", "i", 3),
        (None, "i", r"[0-9]+"),  # No condition
    ],
)
def test_kernel_breakpoint(app: gdb, condition, exp_var, exp_val):
    """Function breakpoints and argument initializing

    Test that it is possible to set conditional breakpoint at the beginning
    of the function and use a function argument in the condition.

    It is important that breakpoint by function name hits at the first line in
    the function body and not at the function definition line.

    Test for https://github.com/numba/numba/issues/7415
    SAT-4449
    """

    app.breakpoint("simple_sum.py:13", condition=condition)
    app.run("simple_sum.py")
    app.expect_hit_breakpoint("simple_sum.py:13")
    if exp_var is not None:
        app.print(exp_var, expected=exp_val)
