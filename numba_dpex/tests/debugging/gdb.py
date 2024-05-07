#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

import pytest

import numba_dpex
from numba_dpex.core import config

if config.TESTING_SKIP_NO_DEBUGGING:
    pexpect = pytest.importorskip("pexpect")
else:
    import pexpect

RE_DEVICE_ARRAY = (
    r"\{\s*nitems\s+=\s+[0-9]+,\s+"
    r"itemsize\s+=\s+[0-9]+,\s+"
    r"data\s+=\s+0x[0-9a-f]+,\s+"
    r"shape\s+=\s+\{\s*[0-9]+\s*\},\s+"
    r"strides\s+=\s+\{\s*[0-9]+\s*\}\s*}"
)

RE_DEVICE_ARRAY_TYPE = (
    r"DpnpNdArray\("
    r"dtype=[a-z0-9]+,\s+"
    r"ndim=[0-9]+,\s+"
    r"layout=[A-Z],\s+"
    r"address_space=[0-4],\s+"
    r"usm_type=[a-z]+,\s+"
    r"device=[a-z\_:0-9]+,\s+"
    r"sycl_queue=[A-Za-z:0-9\s\_:]+\)"
)


class gdb:
    def __init__(self):
        self.spawn()
        self.setup_gdb()

    def spawn(self):
        env = os.environ.copy()
        env["NUMBA_OPT"] = "0"
        env["NUMBA_DPEX_OPT"] = "0"
        env["NUMBA_EXTEND_VARIABLE_LIFETIMES"] = "1"
        env["NUMBA_DPEX_DEBUGINFO"] = "1"
        env["NUMBA_DEBUGINFO"] = "1"

        self.child = pexpect.spawn(
            "gdb-oneapi -q python", env=env, encoding="utf-8", timeout=60
        )
        if config.TESTING_LOG_DEBUGGING:
            self.child.logfile = sys.stdout

    def setup_gdb(self):
        self._command("set breakpoint pending on")
        self._command("set style enabled off")  # disable colors symbols

    def teardown_gdb(self):
        self.child.sendintr()
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline("quit")
        # We want to force quit only if program did not do it itself.
        index = self.child.expect(["Quit anyway?", pexpect.EOF], timeout=5)
        if index == 0:
            self.child.sendline("y")

    def _command(self, command):
        self.child.expect("(gdb)", timeout=5)
        self.child.sendline(command)

    def set_environment(self, varname, value):
        self._command(f"set environment {varname} {value}")

    def breakpoint(
        self,
        location: str,
        condition: str = None,
        expected_location=None,
        expected_line: int = None,
    ):
        cmd = f"break {location}"
        if condition is not None:
            cmd += f" if {condition}"
        self._command(cmd)

        if expected_location is not None:
            self.child.expect(
                rf"Thread .* hit Breakpoint .* at {expected_location}"
            )

        if expected_line is not None:
            self.child.expect(f"{expected_line}")

    def run(self, script):
        self._command("run " + self.script_path(script))

    def expect(self, pattern, with_eol=False, **kw):
        self.child.expect(pattern, **kw)
        if with_eol:
            self.expect_eol()

    def expect_eol(self):
        self.child.expect(r"[^\n]*\n")

    def expect_hit_breakpoint(self, expected_location=None):
        expect = r"Breakpoint [0-9\.]+"
        if expected_location is not None:
            # function name + args could be long, so we have to assume that
            # the message may be splitted in multiple lines. It potentially can
            # cause messed up buffer reading, but it must be extremely rare.
            expect += f".* at {expected_location}"
        self.child.expect(expect)
        self.expect_eol()

    def backtrace(self):
        self._command("backtrace")

    def print(self, var, expected=None):
        self._command("print " + var)
        if expected is not None:
            self.child.expect(rf"\$[0-9]+ = {expected}")
            self.expect_eol()

    def set_variable(self, var, val):
        self._command(f"set variable {var} = {val}")

    def info_args(self):
        self._command("info args")

    def info_functions(self, function):
        self._command("info functions " + function)

    def info_locals(self):
        self._command("info locals")

    def next(self):
        self._command("next")

    def nexti(self):
        self._command("nexti")

    def ptype(self, var):
        self._command("ptype " + var)

    def whatis(self, var):
        self._command("whatis " + var)

    def step(self):
        self._command("step")

    def stepi(self):
        self._command("stepi")

    def set_scheduler_lock(self):
        self._command("set scheduler-locking step")

    @staticmethod
    def script_path(script):
        return script_path(script)


def script_path(script):
    package_path = pathlib.Path(numba_dpex.__file__).parent
    return str(package_path / "examples/debug" / script)
