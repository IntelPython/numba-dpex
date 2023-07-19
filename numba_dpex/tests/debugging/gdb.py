#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

from numba_dpex.core import config

from .common import script_path

if config.TESTING_SKIP_NO_DEBUGGING:
    pexpect = pytest.importorskip("pexpect")
else:
    import pexpect


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
        env["NUMBA_DPEX_DEBUGINFO"] = "1"

        self.child = pexpect.spawn(
            "gdb-oneapi -q python", env=env, encoding="utf-8"
        )
        if config.TESTING_LOG_DEBUGGING:
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

    def set_environment(self, varname, value):
        self._command(f"set environment {varname} {value}")

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
        return script_path(script)
