#! /usr/bin/env python
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for Setting Breakpoints

https://www.sourceware.org/gdb/onlinedocs/gdb/Set-Breaks.html
"""


import pytest

from numba_dppy.tests._helper import skip_no_gdb, skip_no_numba055

from .common import breakpoint_by_mark

pytestmark = skip_no_gdb


side_by_side_breakpoint = breakpoint_by_mark(
    "side-by-side.py", "Set breakpoint here"
)

simple_sum_function_breakpoint = breakpoint_by_mark(
    "simple_sum.py", "Function breakpoint location"
)
simple_sum_condition_breakpoint = breakpoint_by_mark(
    "simple_sum.py", "Condition breakpoint location"
)

common_loop_body_native_function_name = {
    "numba": "common_loop_body_242",
    "numba-dppy-kernel": "common_loop_body",
}

breakpoint_api_cases = [
    (side_by_side_breakpoint, "numba"),
    (side_by_side_breakpoint, "numba-dppy-kernel"),
    *((fn, api) for api, fn in common_loop_body_native_function_name.items()),
    *(
        (f"side-by-side.py:{fn}", api)
        for api, fn in common_loop_body_native_function_name.items()
    ),
]


@skip_no_numba055
@pytest.mark.parametrize("breakpoint, api", breakpoint_api_cases)
def test_breakpoint_with_condition_by_function_argument(app, breakpoint, api):
    """Function breakpoints and argument initializing

    Test that it is possible to set conditional breakpoint at the beginning
    of the function and use a function argument in the condition.

    Test for https://github.com/numba/numba/issues/7415
    SAT-4449
    """
    variable_name = "param_a"
    variable_value = "3"
    condition = f"{variable_name} == {variable_value}"

    app.breakpoint(f"{breakpoint} if {condition}")
    app.run(f"side-by-side.py --api={api}")

    app.child.expect(
        fr"Thread .* hit Breakpoint .* at {side_by_side_breakpoint}"
    )

    app.print(variable_name)

    app.child.expect(fr"\$1 = {variable_value}")


@pytest.mark.parametrize(
    "breakpoint, script, expected_location, expected_line",
    [
        # location specified by file name and function name
        # commands/break_file_func
        (
            "simple_sum.py:data_parallel_sum",
            "simple_sum.py",
            simple_sum_function_breakpoint,
            r"23\s+i = dppy.get_global_id\(0\)",
        ),
        # location specified by function name
        # commands/break_func
        (
            "data_parallel_sum",
            "simple_sum.py",
            simple_sum_function_breakpoint,
            r"23\s+i = dppy.get_global_id\(0\)",
        ),
        # location specified by file name and nested function name
        # commands/break_nested_func
        (
            "simple_dppy_func.py:func_sum",
            "simple_dppy_func.py",
            "simple_dppy_func.py:23",
            r"23\s+result = a_in_func \+ b_in_func",
        ),
    ],
)
def test_breakpoint_common(
    app, breakpoint, script, expected_location, expected_line
):
    """Set a breakpoint in the given script."""

    app.breakpoint(breakpoint)
    app.run(script)

    app.child.expect(fr"Thread .* hit Breakpoint .* at {expected_location}")
    app.child.expect(expected_line)


@pytest.mark.parametrize(
    "breakpoint, script, expected_location, expected_line, variable_name, variable_value",
    [
        # commands/break_conditional
        (
            f"{simple_sum_condition_breakpoint} if i == 1",
            "simple_sum.py",
            simple_sum_condition_breakpoint,
            r"24\s+c\[i\] = a\[i\] \+ b\[i\]",
            "i",
            "1",
        )
    ],
)
def test_breakpoint_with_condition_common(
    app,
    breakpoint,
    script,
    expected_location,
    expected_line,
    variable_name,
    variable_value,
):
    """Set a breakpoint with condition and check value of variable."""

    app.breakpoint(breakpoint)
    app.run(script)

    app.child.expect(fr"Thread .* hit Breakpoint .* at {expected_location}")
    app.child.expect(expected_line)

    app.print(variable_name)

    app.child.expect(fr"\$1 = {variable_value}")
