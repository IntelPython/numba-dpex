#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Information About a Frame

https://www.sourceware.org/gdb/onlinedocs/gdb/Frame-Info.html
"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .gdb import RE_DEVICE_ARRAY, RE_DEVICE_ARRAY_TYPE, gdb

pytestmark = skip_no_gdb


@pytest.mark.parametrize(
    "breakpoint, script, expected_line, expected_args",
    [
        (
            "simple_dpex_func.py:18",
            "simple_dpex_func.py",
            r"18\s+i = item\.get_id\(0\)",
            [
                (
                    "a_in_kernel",
                    RE_DEVICE_ARRAY,
                    "type = struct " + RE_DEVICE_ARRAY_TYPE,
                    "type = " + RE_DEVICE_ARRAY_TYPE,
                ),
                (
                    "b_in_kernel",
                    RE_DEVICE_ARRAY,
                    "type = struct " + RE_DEVICE_ARRAY_TYPE,
                    "type = " + RE_DEVICE_ARRAY_TYPE,
                ),
                (
                    "c_in_kernel",
                    RE_DEVICE_ARRAY,
                    "type = struct " + RE_DEVICE_ARRAY_TYPE,
                    "type = " + RE_DEVICE_ARRAY_TYPE,
                ),
            ],
        ),
        (
            "side-by-side.py:15",
            "side-by-side.py --api=numba",
            r"15\s+param_c = param_a \+ numba\.float32\(10\)",
            [
                ("param_a", r"[0-9]+", r"type = float32", r"type = float32"),
                ("param_b", r"[0-9]+", r"type = float32", r"type = float32"),
            ],
        ),
        (
            "side-by-side.py:15",
            "side-by-side.py --api=numba-ndpx-kernel",
            r"15\s+param_c = param_a \+ numba\.float32\(10\)",
            [
                ("param_a", r"[0-9]+", r"type = float32", r"type = float32"),
                ("param_b", r"[0-9]+", r"type = float32", r"type = float32"),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "kind",
    [
        "print",
        "info",
    ],
)
def test_info_args(
    app: gdb, breakpoint, script, expected_line, expected_args, kind
):
    app.breakpoint(breakpoint)
    app.run(script)
    app.expect_hit_breakpoint(expected_location=breakpoint)
    app.expect(expected_line, with_eol=True)

    if kind == "info":
        app.info_args()
        for var, val, _, _ in expected_args:
            app.expect(f"{var} = {val}", with_eol=True)
    else:
        for var, val, exp_ptype, exp_whatis in expected_args:
            app.print(var, expected=val)

            app.ptype(var)
            app.expect(exp_ptype)

            app.whatis(var)
            app.expect(exp_whatis)


def test_info_functions(app):
    app.breakpoint("simple_sum.py:12")
    app.run("simple_sum.py")
    app.expect_hit_breakpoint(expected_location="simple_sum.py:12")
    app.expect(r"12\s+i = item.get_id\(0\)", with_eol=True)

    app.info_functions("data_parallel_sum")

    app.child.expect(r"11:\s+[a-z 0-9\*]+data_parallel_sum")


@pytest.mark.parametrize(
    "api",
    [
        "numba",
        "numba-ndpx-kernel",
    ],
)
def test_print_array_element(app, api):
    """Test access to array elements"""

    app.breakpoint("side-by-side-2.py:17", condition="param_a == 5")
    app.run(f"side-by-side-2.py --api={api}")
    app.expect_hit_breakpoint(expected_location="side-by-side-2.py:17")

    # We can access only c_array, not python style array
    app.print("b.data[5]", 5)


@pytest.mark.parametrize(
    "api",
    [
        "numba",
        "numba-ndpx-kernel",
    ],
)
@pytest.mark.parametrize(
    "assign",
    [
        "print",
        "set_variable",
    ],
)
def test_assignment_to_variable(app, api, assign):
    app.breakpoint("side-by-side-2.py:17", condition="param_a == 5")
    app.run(f"side-by-side-2.py --api={api}")
    app.expect_hit_breakpoint(expected_location="side-by-side-2.py:17")

    app.print("param_a", expected=5)
    if assign == "print":
        app.print("param_a=15")
    else:
        app.set_variable("param_a", 15)
    app.print("param_a", expected=15)

    # Check that we updated actual value, not gdb environment variable
    app.next()
    app.print("param_c", expected=25)
