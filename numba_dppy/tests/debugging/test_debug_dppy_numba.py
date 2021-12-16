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

import shutil

import pytest

from numba_dppy.tests._helper import skip_no_numba055

pytestmark = pytest.mark.skipif(
    not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
)


@pytest.mark.parametrize("api", ["numba", "numba-dppy-kernel"])
def test_breakpoint_row_number(app, api):
    """Test for checking numba and numba-dppy debugging side-by-side."""

    app.breakpoint("side-by-side.py:25")
    app.run("side-by-side.py --api={api}".format(api=api))

    app.child.expect(r"Breakpoint .* at side-by-side.py:25")
    app.child.expect(r"25\s+param_c = param_a \+ 10")


# commands/backtrace
def test_backtrace(app):
    app.breakpoint("simple_dppy_func.py:23")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")

    app.backtrace()

    app.child.expect(r"#0.*__main__::func_sum .* at simple_dppy_func.py:23")
    app.child.expect(r"#1.*__main__::kernel_sum .* at simple_dppy_func.py:30")


# commands/break_conditional
def test_break_conditional(app):
    app.breakpoint("simple_sum.py:24 if i == 1")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:24")
    app.child.expect(r"24\s+c\[i\] = a\[i\] \+ b\[i\]")

    app.print("i")

    app.child.expect(r"\$1 = 1")


@skip_no_numba055
@pytest.mark.parametrize(
    "breakpoint, api",
    [
        ("side-by-side.py:25", "numba"),
        ("side-by-side.py:25", "numba-dppy-kernel"),
        ("common_loop_body_242", "numba"),
        ("common_loop_body", "numba-dppy-kernel"),
    ],
)
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

    app.child.expect(r"Thread .* hit Breakpoint .* at side-by-side.py:25")

    app.print(variable_name)

    app.child.expect(fr"\$1 = {variable_value}")


# commands/break_file_func
def test_break_file_function(app):
    app.breakpoint("simple_sum.py:data_parallel_sum")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:23")
    app.child.expect(r"23\s+i = dppy.get_global_id\(0\)")


# commands/break_func
def test_break_function(app):
    app.breakpoint("data_parallel_sum")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:23")
    app.child.expect(r"23\s+i = dppy.get_global_id\(0\)")


# commands/break_nested_func
def test_break_nested_function(app):
    app.breakpoint("simple_dppy_func.py:func_sum")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")


@skip_no_numba055
def test_info_args(app):
    app.breakpoint("simple_dppy_func.py:29")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:29")
    app.child.expect(r"29\s+i = dppy.get_global_id\(0\)")

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
    app.breakpoint("simple_sum.py:23")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:23")
    app.child.expect(r"23\s+i = dppy.get_global_id\(0\)")

    app.info_functions("data_parallel_sum")

    app.child.expect(r"22:\s+.*__main__::data_parallel_sum\(.*\)")


# commands/local_variables_0
@skip_no_numba055
def test_local_variables(app):
    app.breakpoint("sum_local_vars.py:26")
    app.run("sum_local_vars.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at sum_local_vars.py:26")
    app.child.expect(r"26\s+c\[i\] = l1 \+ l2")

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


# commands/next
def test_next(app):
    app.breakpoint("simple_dppy_func.py:30")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:30")
    app.child.expect(
        r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.next()
    app.next()

    app.child.expect(r"Done\.\.\.")


# commands/step_dppy_func
def test_step(app):
    app.breakpoint("simple_dppy_func.py:30")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:30")
    app.child.expect(
        r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.step()
    app.step()

    app.child.expect(r"__main__::func_sum \(.*\) at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")


# commands/stepi
def test_stepi(app):
    app.breakpoint("simple_dppy_func.py:30")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:30")
    app.child.expect(
        r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.stepi()

    app.child.expect(
        r"0x[0-f]+\s+30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.stepi()

    app.child.expect(r"Switching to Thread")
    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:30")
    app.child.expect(
        r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
