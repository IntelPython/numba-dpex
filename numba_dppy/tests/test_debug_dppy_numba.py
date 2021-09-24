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

pexpect = pytest.importorskip("pexpect")

pytestmark = pytest.mark.skipif(
    not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
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

        self.child = pexpect.spawn("gdb-oneapi -q python", env=env, encoding="utf-8")
        # self.child.logfile = sys.stdout

    def setup_gdb(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("set breakpoint pending on")
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("set style enabled off")  # disable colors symbols

    def teardown_gdb(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("quit")
        self.child.expect("Quit anyway?", timeout=5)
        self.child.sendline("y")

    def breakpoint(self, breakpoint):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("break " + breakpoint)

    def run(self, script):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("run " + self.script_path(script))

    def backtrace(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("backtrace")

    def print(self, var):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("print " + var)

    def info_functions(self, function):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("info functions " + function)

    def info_locals(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("info locals")

    def next(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("next")

    def ptype(self, var):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("ptype " + var)

    def whatis(self, var):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("whatis " + var)

    def step(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("step")

    def stepi(self):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("stepi")

    @staticmethod
    def script_path(script):
        package_path = pathlib.Path(numba_dppy.__file__).parent
        return str(package_path / "examples/debug" / script)


@pytest.mark.parametrize("api", ["numba", "numba-dppy"])
def test_breakpoint_row_number(api):
    app = gdb()

    app.breakpoint("dppy_numba_basic.py:25")
    app.run("dppy_numba_basic.py --api={api}".format(api=api))

    app.child.expect(r"Thread .* hit Breakpoint .* at dppy_numba_basic.py:25")
    app.child.expect(r"25\s+param_c = param_a \+ 10")


# commands/backtrace
def test_backtrace():
    app = gdb()

    app.breakpoint("simple_dppy_func.py:23")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")

    app.backtrace()

    app.child.expect(r"#0.*__main__::func_sum .* at simple_dppy_func.py:23")
    app.child.expect(r"#1.*__main__::kernel_sum .* at simple_dppy_func.py:30")


# commands/break_conditional
def test_break_conditional():
    app = gdb()

    app.breakpoint("simple_sum.py:24 if i == 1")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:24")
    app.child.expect(r"24\s+c\[i\] = a\[i\] \+ b\[i\]")

    app.print("i")

    app.child.expect(r"\$1 = 1")


# commands/break_file_func
def test_break_file_function():
    app = gdb()

    app.breakpoint("simple_sum.py:data_parallel_sum")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:23")
    app.child.expect(r"23\s*i = dppy.get_global_id\(0\)")


# commands/break_func
def test_break_function():
    app = gdb()

    app.breakpoint("data_parallel_sum")
    app.run("simple_sum.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_sum.py:20")
    app.child.expect(r"20\s+@dppy\.kernel\(debug=True\)")


# commands/break_nested_func
def test_break_nested_function():
    app = gdb()

    app.breakpoint("simple_dppy_func.py:func_sum")
    app.run("simple_dppy_func.py")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:22")
    app.child.expect(r"22\s+result = a_in_func \+ b_in_func")


# commands/info_func
def test_info_functions():
    app = gdb()

    app.breakpoint("simple_sum.py:22")
    app.run("simple_sum.py")
    app.info_functions("data_parallel_sum")

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:22")
    app.child.expect(r"22\s+result = a_in_func \+ b_in_func")
    app.child.expect(
        r"20:\s+void __main__::data_parallel_sum\(DPPYArray\<float, 1, C, mutable, aligned\>, .*\);"
    )


# commands/local_variables_0
def test_local_variables():
    app = gdb()

    app.breakpoint("sum_local_vars.py:22")
    app.run("sum_local_vars.py")
    app.info_locals()
    app.next()
    app.next()
    app.next()
    app.next()
    app.print("a")
    app.print("l1")
    app.print("l2")
    app.ptype("a")
    app.whatis("a")
    app.ptype("l1")
    app.whatis("l1")

    app.child.expect(r"Thread .* hit Breakpoint .* at sum_local_vars.py:22")
    app.child.expect(r"22\s+i = dppy.get_global_id\(0\)")
    app.child.expect(r"a = '\\000' \<repeats .* times\>")
    app.child.expect(r"b = '\\000' \<repeats .* times\>")
    app.child.expect(r"c = '\\000' \<repeats .* times\>")
    app.child.expect(r"i = .*")
    app.child.expect(r"l1 = .*")
    app.child.expect(r"l2 = .*")
    app.child.expect(r"__ocl_dbg_gid0 = .*")
    app.child.expect(r"__ocl_dbg_gid1 = .*")
    app.child.expect(r"__ocl_dbg_gid2 = .*")
    app.child.expect(r"__ocl_dbg_lid0 = .*")
    app.child.expect(r"__ocl_dbg_lid1 = .*")
    app.child.expect(r"__ocl_dbg_lid2 = .*")
    app.child.expect(r"__ocl_dbg_grid0 = .*")
    app.child.expect(r"__ocl_dbg_grid1 = .*")
    app.child.expect(r"__ocl_dbg_grid2 = .*")
    app.child.expect(r"\$1 = '\\000' \<repeats 55 times\>")
    app.child.expect(r"\$3 = 2.5931931659579277")
    app.child.expect(r"\$4 = 0.22954882979393004")
    app.child.expect(r"type = byte \[56\]")
    app.child.expect(r"type = double")


# commands/next
def test_next():
    app = gdb()

    app.breakpoint("simple_dppy_func.py:29")
    app.run("simple_dppy_func.py")
    app.next()
    app.next()

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:29")
    app.child.expect(
        r"29\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
    app.child.expect(r"Done\.\.\.")


# commands/step_dppy_func
def test_step():
    app = gdb()

    app.breakpoint("simple_dppy_func.py:29")
    app.run("simple_dppy_func.py")
    app.step()
    app.step()

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:29")
    app.child.expect(
        r"29\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
    app.child.expect(r"__main__::func_sum \(\) at simple_dppy_func.py:22")
    app.child.expect(r"22\s+result = a_in_func \+ b_in_func")


# commands/stepi
def test_stepi():
    app = gdb()

    app.breakpoint("simple_dppy_func.py:29")
    app.run("simple_dppy_func.py")
    app.stepi()

    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dppy_func.py:29")
    app.child.expect(
        r"29\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
    app.child.expect(
        r".+29\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\])"
    )
