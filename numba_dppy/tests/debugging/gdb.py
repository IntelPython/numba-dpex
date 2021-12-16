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
import sys

import pytest

import numba_dppy
from numba_dppy import config

pexpect = pytest.importorskip("pexpect")


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
