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

import pytest

from numba_dppy.tests._helper import skip_no_gdb, skip_no_numba055

pytestmark = skip_no_gdb


# commands/backtrace
def test_backtrace(app):
    app.breakpoint("simple_dppy_func.py:23")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")

    app.backtrace()

    app.child.expect(r"#0.*__main__::func_sum .* at simple_dppy_func.py:23")
    app.child.expect(r"#1.*__main__::kernel_sum .* at simple_dppy_func.py:30")


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
