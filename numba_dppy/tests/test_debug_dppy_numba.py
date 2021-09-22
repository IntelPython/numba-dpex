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

from numba_dppy.tests._helper import run_debug_command, make_check


@pytest.mark.skipif(
    not shutil.which("gdb-oneapi"),
    reason="Intel Distribution for GDB is not available",
)
def test_breakpoint_row_number():
    """
    break dppy_numba.py:24
    run dppy_numba.py --dppy
    """

    ref_output = [
        r"24.+param_c = param_a \+ 10",
    ]

    numba_ref_test = True
    dppy_ref_test = True

    with run_debug_command("dppy_numba_jit") as command_out:
        for ref in ref_output:
            numba_ref_test &= make_check(command_out, ref)

    with run_debug_command("dppy_numba_kernel") as command_out:
        for ref in ref_output:
            dppy_ref_test &= make_check(command_out, ref)

    assert numba_ref_test and dppy_ref_test
