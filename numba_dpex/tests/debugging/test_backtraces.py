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
"""Tests for Backtraces

https://www.sourceware.org/gdb/onlinedocs/gdb/Backtrace.html
"""

from numba_dpex.tests._helper import skip_no_gdb

from .common import setup_breakpoint

pytestmark = skip_no_gdb


def test_backtrace(app):
    """Simple test for backtrace.

    commands/backtrace
    """
    setup_breakpoint(
        app,
        "simple_dppy_func.py:23",
        expected_line=r"23\s+result = a_in_func \+ b_in_func",
    )

    app.backtrace()

    app.child.expect(r"#0.*__main__::func_sum .* at simple_dppy_func.py:23")
    app.child.expect(r"#1.*__main__::kernel_sum .* at simple_dppy_func.py:30")
