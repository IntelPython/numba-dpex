#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Backtraces

https://www.sourceware.org/gdb/onlinedocs/gdb/Backtrace.html
"""
import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .common import setup_breakpoint

pytestmark = skip_no_gdb


@pytest.mark.xfail  # TODO: https://github.com/IntelPython/numba-dpex/issues/1216
def test_backtrace(app):
    """Simple test for backtrace.

    commands/backtrace
    """
    setup_breakpoint(
        app,
        "simple_dpex_func.py:13",
        expected_line=r"13\s+result = a_in_func \+ b_in_func",
    )

    app.backtrace()

    app.child.expect(r"#0.*__main__::func_sum .* at simple_dpex_func.py:13")


#    app.child.expect(r"#1.*__main__::kernel_sum .* at simple_dpex_func.py:20")
