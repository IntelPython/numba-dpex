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
"""Tests for Information About a Frame

https://www.sourceware.org/gdb/onlinedocs/gdb/Frame-Info.html
"""

import pytest

from numba_dppy.tests._helper import skip_no_gdb, skip_no_numba055

from .common import setup_breakpoint
from .test_breakpoints import side_by_side_breakpoint

pytestmark = skip_no_gdb


def side_by_side_case(api):
    return (
        side_by_side_breakpoint,
        f"side-by-side.py --api={api}",
        None,
        (
            r"param_a = 0",
            r"param_b = 0",
        ),
        ("param_a", r"\$1 = 0"),
        ("param_a", r"type = float32"),
        ("param_a", r"type = float32"),
    )


@skip_no_numba055
@pytest.mark.parametrize(
    "breakpoint, script, expected_line, expected_args, expected_print, expected_ptype, expected_whatis",
    [
        (
            "simple_dppy_func.py:29",
            "simple_dppy_func.py",
            r"29\s+i = dppy.get_global_id\(0\)",
            (
                r"a_in_kernel = {meminfo = ",
                r"b_in_kernel = {meminfo = ",
                r"c_in_kernel = {meminfo = ",
            ),
            ("a_in_kernel", r"\$1 = {meminfo = "),
            ("a_in_kernel", r"type = struct array\(float32, 1d, C\).*}\)"),
            ("a_in_kernel", r"type = array\(float32, 1d, C\) \({.*}\)"),
        ),
        side_by_side_case("numba"),
        side_by_side_case("numba-dppy-kernel"),
    ],
)
def test_info_args(
    app,
    breakpoint,
    script,
    expected_line,
    expected_args,
    expected_print,
    expected_ptype,
    expected_whatis,
):
    """Test for info args command.

    SAT-4462
    Issue: https://github.com/numba/numba/issues/7414
    Fix: https://github.com/numba/numba/pull/7177
    """

    setup_breakpoint(app, breakpoint, script, expected_line=expected_line)

    app.info_args()

    for arg in expected_args:
        app.child.expect(arg)

    app.print(expected_print[0])
    app.child.expect(expected_print[1])

    app.ptype(expected_ptype[0])
    app.child.expect(expected_ptype[1])

    app.whatis(expected_whatis[0])
    app.child.expect(expected_whatis[1])


# commands/info_func
@skip_no_numba055
def test_info_functions(app):
    expected_line = r"23\s+i = dppy.get_global_id\(0\)"
    setup_breakpoint(app, "simple_sum.py:23", expected_line=expected_line)

    app.info_functions("data_parallel_sum")

    app.child.expect(r"22:\s+.*__main__::data_parallel_sum\(.*\)")


# commands/local_variables_0
@skip_no_numba055
def test_local_variables(app):
    expected_line = r"26\s+c\[i\] = l1 \+ l2"
    setup_breakpoint(app, "sum_local_vars.py:26", expected_line=expected_line)

    app.info_locals()

    app.child.expect(r"i = 0")
    app.child.expect(r"l1 = [0-9]\.[0-9]{3}")
    app.child.expect(r"l2 = [0-9]\.[0-9]{3}")

    app.print("a")
    app.child.expect(r"\$1 = {meminfo = ")

    app.print("l1")
    app.child.expect(r"\$2 = [0-9]\.[0-9]{3}")

    app.print("l2")
    app.child.expect(r"\$3 = [0-9]\.[0-9]{3}")

    app.ptype("a")
    app.child.expect(r"type = struct array\(float32, 1d, C\).*}\)")

    app.whatis("a")
    app.child.expect(r"type = array\(float32, 1d, C\) \({.*}\)")

    app.ptype("l1")
    app.child.expect(r"type = float64")

    app.whatis("l1")
    app.child.expect(r"type = float64")
