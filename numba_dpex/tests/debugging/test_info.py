#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Information About a Frame

https://www.sourceware.org/gdb/onlinedocs/gdb/Frame-Info.html
"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .common import setup_breakpoint
from .test_breakpoints import side_by_side_breakpoint

pytestmark = skip_no_gdb


def side_by_side_info_case(api):
    return (
        side_by_side_breakpoint,
        f"side-by-side.py --api={api}",
        None,
        (
            r"param_a = .*",
            r"param_b = .*",
        ),
    )


def side_by_side_print_case(api):
    return (
        side_by_side_breakpoint,
        f"side-by-side.py --api={api}",
        None,
        (
            "param_a",
            r"\$1 = .*",
            r"type = float32",
            r"type = float32",
        ),
    )


@pytest.mark.parametrize(
    "breakpoint, script, expected_line, expected_args",
    [
        (
            "simple_dpex_func.py:18",
            "simple_dpex_func.py",
            r".*18.*i = ndpx.get_global_id\(0\)",
            (
                r"a_in_kernel = {meminfo = ",
                r"b_in_kernel = {meminfo = ",
                r"c_in_kernel = {meminfo = ",
            ),
        ),
        side_by_side_info_case("numba"),
        side_by_side_info_case("numba-ndpx-kernel"),
    ],
)
def test_info_args(app, breakpoint, script, expected_line, expected_args):
    setup_breakpoint(app, breakpoint, script, expected_line=expected_line)

    app.info_args()

    app.child.expect(list(expected_args))


@pytest.mark.parametrize(
    "breakpoint, script, expected_line, expected_info",
    [
        (
            "simple_dpex_func.py:18",
            "simple_dpex_func.py",
            r".*18.*i = ndpx.get_global_id\(0\)",
            (
                "a_in_kernel",
                r"\$1 = {meminfo = ",
                r"type = struct DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)",
                r"type = DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)",
            ),
        ),
        side_by_side_print_case("numba"),
        side_by_side_print_case("numba-ndpx-kernel"),
    ],
)
def test_print_args(app, breakpoint, script, expected_line, expected_info):
    setup_breakpoint(app, breakpoint, script, expected_line=expected_line)

    variable, expected_print, expected_ptype, expected_whatis = expected_info

    app.print(variable)
    app.child.expect(expected_print)

    app.ptype(variable)
    app.child.expect(expected_ptype)

    app.whatis(variable)
    app.child.expect(expected_whatis)


def test_info_functions(app):
    expected_line = r"13\s+i = ndpx.get_global_id\(0\)"
    setup_breakpoint(app, "simple_sum.py:13", expected_line=expected_line)

    app.info_functions("data_parallel_sum")

    app.child.expect(r"12:\s+.*__main__::data_parallel_sum.*\(.*\)")


def side_by_side_info_locals_case(api):
    return (
        {},
        "side-by-side.py:16 if param_a == 5",
        f"side-by-side.py --api={api}",
        None,
        (
            r"param_c = 15",
            r"param_d = 0",
            r"result = 0",
        ),
        (),
    )


def side_by_side_2_info_locals_case(api):
    if api == "numba":
        ptype = r"type = struct array\(float32, 1d, C\) \({.*}\)"
        whatis = r"type = array\(float32, 1d, C\) \({.*}\)"
    elif api == "numba-ndpx-kernel":
        ptype = r"type = struct DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)"
        whatis = (
            r"type = DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)"
        )

    return (
        {},
        "side-by-side-2.py:18 if param_a == 5",
        f"side-by-side-2.py --api={api}",
        None,
        (
            r"param_a = 5",
            r"param_b = 5",
            r"param_c = .*",
            r"param_d = .*",
            r"result = 0",
        ),
        ((r"a", r"\$1 = {meminfo = ", ptype, whatis),),
    )


@pytest.mark.parametrize(
    "env, breakpoint, script, expected_line, expected_info_locals, expected_info",
    [
        (
            {"NUMBA_DPEX_OPT": 0},
            "sum_local_vars.py:16",
            "sum_local_vars.py",
            r"16\s+c\[i\] = l1 \+ l2",
            (
                r"i = .*",
                r"l1 = [0-9]\.[0-9]*.*",
                r"l2 = [0-9]\.[0-9]*.*",
            ),
            (
                (
                    "a",
                    r"\$1 = {meminfo = ",
                    r"type = struct DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)",
                    r"type = DpnpNdArray\(dtype=float32, ndim=1, layout=C.*\) \({.*}\)",
                ),
                (
                    "l1",
                    r"\$2 = [0-9]\.[0-9]*",
                    r"type = float64",
                    r"type = float64",
                ),
                (
                    "l2",
                    r"\$3 = [0-9]\.[0-9]*",
                    r"type = float64",
                    r"type = float64",
                ),
            ),
        ),
        (
            {"NUMBA__DPEX_OPT": 1},
            "sum_local_vars.py:16",
            "sum_local_vars.py",
            r"16\s+c\[i\] = l1 \+ l2",
            (
                r".*i = [0-9]",
                r"l1 = [0-9]\.[0-9]*",
                r"l2 = [0-9]\.[0-9]*",
            ),
            (
                (
                    "i",
                    r"\$1 = [0-9]",
                    r"type = int64",
                    r"type = int64",
                ),
                (
                    "l1",
                    r"\$2 = [0-9]\.[0-9]*",
                    r"type = float64",
                    r"type = float64",
                ),
                (
                    "l1",
                    r"\$3 = [0-9]\.[0-9]*",
                    r"type = float64",
                    r"type = float64",
                ),
            ),
        ),
        (
            {"NUMBA_EXTEND_VARIABLE_LIFETIMES": 1},
            "side-by-side.py:18",
            "side-by-side.py --api=numba-ndpx-kernel",
            r"18\s+result = param_c \+ param_d",
            (r"param_c = [0-9]*", r"param_d = [0-9]*", r"result = [0-9]*"),
            (),
        ),
        (
            {"NUMBA_EXTEND_VARIABLE_LIFETIMES": 0},
            "side-by-side.py:18",
            "side-by-side.py --api=numba-ndpx-kernel",
            r"18\s+result = param_c \+ param_d",
            (r"param_c = [0-9]*", r"param_d = [0-9]*", r"result = [0-9]*"),
            (),
        ),
        side_by_side_info_locals_case("numba"),
        side_by_side_info_locals_case("numba-ndpx-kernel"),
        side_by_side_2_info_locals_case("numba"),
        side_by_side_2_info_locals_case("numba-ndpx-kernel"),
    ],
)
def test_info_locals(
    app,
    env,
    breakpoint,
    script,
    expected_line,
    expected_info_locals,
    expected_info,
):
    """Test info locals with different environment variables.

    commands/local_variables_0
    commands/local_variables_1

    SAT-4454
    Provide information about variables (arrays).
    Issue: https://github.com/numba/numba/issues/7414
    Fix: https://github.com/numba/numba/pull/7177
         https://github.com/numba/numba/pull/7421
    """

    for varname, value in env.items():
        app.set_environment(varname, value)

    setup_breakpoint(app, breakpoint, script, expected_line=expected_line)

    app.info_locals()

    app.child.expect(list(expected_info_locals))

    for info in expected_info:
        variable, expected_print, expected_ptype, expected_whatis = info

        app.print(variable)
        app.child.expect(expected_print)

        app.ptype(variable)
        app.child.expect(expected_ptype)

        app.whatis(variable)
        app.child.expect(expected_whatis)


def side_by_side_2_print_array_element_case(api):
    return (
        "side-by-side-2.py:17 if param_a == 5",
        f"side-by-side-2.py --api={api}",
        [(r"b.data[5]", r"\$1 = 5")],
    )


@pytest.mark.parametrize(
    "breakpoint, script, expected_info",
    [
        side_by_side_2_print_array_element_case("numba"),
        side_by_side_2_print_array_element_case("numba-ndpx-kernel"),
    ],
)
def test_print_array_element(app, breakpoint, script, expected_info):
    """Test access to array elements"""

    setup_breakpoint(app, breakpoint, script)

    for info in expected_info:
        variable, expected_print = info

        app.print(variable)
        app.child.expect(expected_print)


def side_by_side_2_assignment_to_variable_case(api):
    return (
        "side-by-side-2.py:20 if param_c == 15",
        f"side-by-side-2.py --api={api}",
        [
            (r"param_c", r"\$1 = 15"),
            (r"param_c=150", r"\$2 = 150"),
            (r"param_c", r"\$3 = 150"),
            (r"i", r"\$4 = .*"),
            (r"i=50", r"\$5 = 50"),
            (r"i", r"\$6 = 50"),
        ],
    )


@pytest.mark.parametrize(
    "breakpoint, script, expected_info",
    [
        side_by_side_2_assignment_to_variable_case("numba"),
        side_by_side_2_assignment_to_variable_case("numba-ndpx-kernel"),
    ],
)
def test_assignment_to_variable(app, breakpoint, script, expected_info):
    setup_breakpoint(app, breakpoint, script)

    for info in expected_info:
        variable, expected_print = info

        app.print(variable)
        app.child.expect(expected_print)
