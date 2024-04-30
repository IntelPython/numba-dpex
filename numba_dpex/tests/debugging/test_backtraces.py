#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Backtraces

https://www.sourceware.org/gdb/onlinedocs/gdb/Backtrace.html
"""

from numba_dpex.tests._helper import skip_no_gdb

pytestmark = skip_no_gdb


def test_backtrace(app):
    """Simple test for backtrace.

    commands/backtrace
    """
    app.breakpoint("simple_dpex_func.py:12")
    app.run("simple_dpex_func.py")
    app.expect_hit_breakpoint("simple_dpex_func.py:12")

    app.backtrace()

    app.expect(r"#0.*func_sum.* at simple_dpex_func.py:12", with_eol=True)
    app.expect(r"#1.*kernel_sum", with_eol=True)
