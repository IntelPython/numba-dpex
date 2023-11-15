#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Continuing and Stepping

https://www.sourceware.org/gdb/onlinedocs/gdb/Continuing-and-Stepping.html
"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .common import setup_breakpoint

pytestmark = skip_no_gdb


# commands/next
@pytest.mark.xfail  # TODO: https://github.com/IntelPython/numba-dpex/issues/1216
def test_next(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:20",
        expected_line=r"20\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.next()
    app.next()

    app.child.expect(r"Done\.\.\.")


# commands/step_dpex_func
@pytest.mark.xfail  # TODO: https://github.com/IntelPython/numba-dpex/issues/1216
def test_step(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:20",
        expected_line=r"20\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.step()
    app.step()

    app.child.expect(r"__main__::func_sum \(.*\) at simple_dpex_func.py:13")
    app.child.expect(r"13\s+result = a_in_func \+ b_in_func")


# commands/stepi
@pytest.mark.xfail  # TODO: https://github.com/IntelPython/numba-dpex/issues/1216
def test_stepi(app):
    setup_breakpoint(
        app,
        "simple_dpex_func.py:20",
        expected_line=r"20\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
    )

    app.stepi()

    app.child.expect(
        r"0x[0-f]+\s+20\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )

    app.stepi()

    app.child.expect(r"Switching to Thread")
    app.child.expect(r"Thread .* hit Breakpoint .* at simple_dpex_func.py:20")
    app.child.expect(
        r"20\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)"
    )
