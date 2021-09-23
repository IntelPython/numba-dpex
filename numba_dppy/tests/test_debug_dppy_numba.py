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
    reason="Intel Distribution for GDB is not available",
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

    @staticmethod
    def script_path(script):
        package_path = pathlib.Path(numba_dppy.__file__).parent
        return str(package_path / "examples/debug" / script)


@pytest.mark.parametrize("api", ["numba", "numba-dppy"])
def test_breakpoint_row_number(api):
    app = gdb()

    app.breakpoint("dppy_numba_basic.py:24")
    app.run("dppy_numba_basic.py --api={api}".format(api=api))

    app.child.expect("Thread .* hit Breakpoint .* at dppy_numba_basic.py:24")
    app.child.expect("24\s+param_c = param_a \+ 10")
