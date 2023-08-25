#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Setting Breakpoints

https://www.sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html
"""


import pytest

from numba_dpex.tests._helper import skip_no_gdb, skip_no_numba056

from .examples_common import (
    breakpoint_by_function,
    breakpoint_by_mark,
    setup_breakpoint,
)

pytestmark = skip_no_gdb


side_by_side_breakpoint = breakpoint_by_function(
    "side-by-side.py", "common_loop_body"
)

simple_sum_condition_breakpoint = breakpoint_by_mark(
    "simple_sum.py", "Condition breakpoint location"
)

common_loop_body_native_function_name = {
    "numba": "common_loop_body",
    "numba-dpex-kernel": "common_loop_body",
}

breakpoint_api_cases = [
    (side_by_side_breakpoint, "numba"),
    (side_by_side_breakpoint, "numba-dpex-kernel"),
    *((fn, api) for api, fn in common_loop_body_native_function_name.items()),
    *(
        (f"side-by-side.py:{fn}", api)
        for api, fn in common_loop_body_native_function_name.items()
    ),
]


@skip_no_numba056
@pytest.mark.parametrize("breakpoint, api", breakpoint_api_cases)
def test_breakpoint_with_condition_by_function_argument(app, breakpoint, api):
    """Function breakpoints and argument initializing

    Test that it is possible to set conditional breakpoint at the beginning
    of the function and use a function argument in the condition.

    It is important that breakpoint by function name hits at the firts line in
    the function body and not at the function definition line.

    Test for https://github.com/numba/numba/issues/7415
    SAT-4449
    """
    variable_name = "param_a"
    variable_value = "3"
    condition = f"{variable_name} == {variable_value}"

    app.breakpoint(f"{breakpoint} if {condition}")
    app.run(f"side-by-side.py --api={api}")

    app.child.expect(
        rf"Thread .* hit Breakpoint .* at {side_by_side_breakpoint}"
    )

    app.print(variable_name)

    app.child.expect(rf"\$1 = {variable_value}")


@pytest.mark.parametrize(
    "breakpoint, script",
    [
        # location specified by file name and function name
        # commands/break_file_func
        ("simple_sum.py:data_parallel_sum", None),
        # location specified by function name
        # commands/break_func
        ("data_parallel_sum", "simple_sum.py"),
        # location specified by file name and nested function name
        # commands/break_nested_func
        ("simple_dpex_func.py:func_sum", None),
    ],
)
def test_breakpoint_common(app, breakpoint, script):
    """Set a breakpoint in the given script."""
    setup_breakpoint(app, breakpoint, script=script)


@pytest.mark.parametrize(
    "breakpoint, variable_name, variable_value",
    [
        # commands/break_conditional
        (f"{simple_sum_condition_breakpoint} if i == 1", "i", "1"),
    ],
)
def test_breakpoint_with_condition_common(
    app,
    breakpoint,
    variable_name,
    variable_value,
):
    """Set a breakpoint with condition and check value of variable."""

    setup_breakpoint(app, breakpoint)

    app.print(variable_name)

    app.child.expect(rf"\$1 = {variable_value}")
