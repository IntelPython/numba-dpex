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

from numba_dppy.tests._helper import skip_no_gdb, skip_no_numba055

from .common import setup_breakpoint

pytestmark = skip_no_gdb


@skip_no_numba055
def test_info_args(app):
    """Test for info args command.

    SAT-4462
    """

    expected_line = r"29\s+i = dppy.get_global_id\(0\)"
    setup_breakpoint(app, "simple_dppy_func.py:29", expected_line=expected_line)

    app.info_args()

    app.child.expect(r"a_in_kernel = {meminfo = ")
    app.child.expect(r"b_in_kernel = {meminfo = ")
    app.child.expect(r"c_in_kernel = {meminfo = ")

    app.print("a_in_kernel")
    app.child.expect(r"\$1 = {meminfo = ")

    app.ptype("a_in_kernel")
    app.child.expect(r"type = struct array\(float32, 1d, C\).*}\)")

    app.whatis("a_in_kernel")
    app.child.expect(r"type = array\(float32, 1d, C\) \({.*}\)")


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
