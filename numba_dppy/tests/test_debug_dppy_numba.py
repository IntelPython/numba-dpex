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

import os
import pathlib
import shutil
import sys

import pytest

import numba_dppy
from numba_dppy import config
from numba_dppy.numba_support import numba_version

pexpect = pytest.importorskip("pexpect")

pytestmark = pytest.mark.skipif(
    not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
)


skip_no_numba055 = pytest.mark.skipif(
    numba_version < (0, 55), reason="Need Numba 0.55 or higher"
)


# TODO: go to helper
class gdb:
    def __init__(self):
        self.spawn()
        self.setup_gdb()

    def __del__(self):
        self.teardown_gdb()

    def spawn(self):
        env = os.environ.copy()
        env["NUMBA_OPT"] = "0"
        env["NUMBA_EXTEND_VARIABLE_LIFETIMES"] = "1"

        self.child = pexpect.spawn(
            "gdb-oneapi -q python", env=env, encoding="utf-8"
        )
        if config.DEBUG:
            self.child.logfile = sys.stdout

    def setup_gdb(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("set breakpoint pending on")
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("set style enabled off")  # disable colors symbols

    def teardown_gdb(self):
        self.child.sendintr()
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("quit")
        self.child.expect("Quit anyway?", timeout=5)
        self.child.sendline("y")

    def _command(self, command):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline(command)

    def breakpoint(self, breakpoint):
        self._command("break " + breakpoint)

    def run(self, script):
        self._command("run " + self.script_path(script))

    def backtrace(self):
        self._command("backtrace")

    def print(self, var):
        self._command("print " + var)

    def info_args(self):
        self._command("info args")

    def info_functions(self, function):
        self._command("info functions " + function)

    def info_locals(self):
        self._command("info locals")

    def next(self):
        self._command("next")

    def ptype(self, var):
        self._command("ptype " + var)

    def whatis(self, var):
        self._command("whatis " + var)

    def step(self):
        self._command("step")

    def stepi(self):
        self._command("stepi")

    @staticmethod
    def script_path(script):
        package_path = pathlib.Path(numba_dppy.__file__).parent
        return str(package_path / "examples/debug" / script)


@pytest.fixture
def app():
    return gdb()


@pytest.mark.parametrize("api", ["numba", "numba-dppy"])
def test_breakpoint_row_number(app, api):
    app.breakpoint("dppy_numba_basic.py:25")
    app.run("dppy_numba_basic.py --api={api}".format(api=api))

    app.child.expect(r"Breakpoint .* at dppy_numba_basic.py:25")
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
def test_break_conditional_with_func_arg(app):
    app.breakpoint("simple_dppy_func.py:23 if a_in_func == 3")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")

    app.print("a_in_func")

    app.child.expect(r"\$1 = 3")


@skip_no_numba055
def test_break_conditional_by_func_name_with_func_arg(app):
    app.breakpoint("func_sum if a_in_func == 3")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")

    app.print("a_in_func")

    app.child.expect(r"\$1 = 3")


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
