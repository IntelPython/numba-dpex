#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Continuing and Stepping

https://www.sourceware.org/gdb/onlinedocs/gdb/Continuing-and-Stepping.html
"""

from numba_dpex.tests._helper import skip_no_gdb

from .common import setup_breakpoint

pytestmark = skip_no_gdb


# commands/next
def test_next(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:30",
        expected_line=r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.next()
    app.next()

    app.child.expect(r"Done\.\.\.")


# commands/step_dpex_func
def test_step(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:30",
        expected_line=r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.step()
    app.step()

    app.child.expect(r"__main__::func_sum \(.*\) at simple_dpex_func.py:23")
    app.child.expect(r"23\s+result = a_in_func \+ b_in_func")


# commands/stepi
def test_stepi(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:30",
        expected_line=r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.stepi()

    app.child.expect(
        r"0x[0-f]+\s+30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.stepi()

    app.child.expect(r"Switching to Thread")
    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dpex_func.py:30")
    app.child.expect(
        r"30\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
