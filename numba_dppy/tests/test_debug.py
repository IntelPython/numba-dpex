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

import re
import subprocess
from subprocess import Popen, PIPE
import os
import pytest

import contextlib


def make_check(text, val_to_search):
    m = re.search(val_to_search, text, re.I)
    got = m is not None
    return got


def utils():
    wd = os.getcwd() + "/numba_dppy/examples/debug"
    os.chdir(wd)
    os.environ["NUMBA_OPT"] = "0"


@contextlib.contextmanager
def run_command(command_name):
    previous_dir = os.getcwd()
    previous_numba_opt = os.environ.get("NUMBA_OPT")
    os.chdir(previous_dir + "/numba_dppy/examples/debug")
    os.environ["NUMBA_OPT"] = "0"

    process = Popen(
        [
            "gdb-oneapi",
            "-q",
            "-command",
            command_name,
            "python",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    (output, err) = process.communicate()
    process.wait()
    output_str = str(output)
    yield output_str

    os.chdir(previous_dir)
    if previous_numba_opt:
        os.environ["NUMBA_OPT"] = previous_numba_opt
    else:
        os.environ.pop("NUMBA_OPT")


# out = Popen(["ls", "-la", "."], stdout=PIPE)
def test_backtrace():
    expected_commands = [
        r"Thread .*",
        r"Thread .\.. hit Breakpoint ., with SIMD lanes \[.\-.\], __main__::func_sum \(\) at simple_dppy_func.py:..",
        r".*result = a_in_func \+ b_in_func",
        r"\#0  __main__::func_sum \(\) at simple_dppy_func.py:..",
        r"\#1  __main__::kernel_sum \(\) at simple_dppy_func.py:..",
        r"\[Switching to Thread .* lane .\]",
        r"Thread .\.. hit Breakpoint 1, with SIMD lanes \[.\-.\], __main__::func_sum \(\) at simple_dppy_func.py:..",
        r".*result = a_in_func \+ b_in_func",
        r"Done\.\.\.",
    ]

    with run_command("commands/backtrace") as backtrace_out:
        for command in expected_commands:
            assert make_check(backtrace_out, command)


def test_break_func():
    expected_commands = [
        r"Thread .\.. hit Breakpoint ., with SIMD lanes \[.\-.\], __main__::data_parallel_sum \(\) at simple_sum.py:..",
        r"i = dppy.get_global_id\(0\)",
        r"Done\.\.\.",
    ]

    with run_command("commands/break_func") as break_func_out:
        for command in expected_commands:
            assert make_check(break_func_out, command)
