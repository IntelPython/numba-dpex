#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Continuing and Stepping

https://www.sourceware.org/gdb/onlinedocs/gdb/Continuing-and-Stepping.html
"""

import pytest

from numba_dpex.tests._helper import skip_no_gdb

from .gdb import gdb

pytestmark = skip_no_gdb


def test_next(app: gdb):
    app.breakpoint("simple_dpex_func.py:18")
    app.run("simple_dpex_func.py")
    app.expect_hit_breakpoint(expected_location="simple_dpex_func.py:18")
    app.expect(r"18\s+i = item.get_id\(0\)", with_eol=True)
    app.set_scheduler_lock()
    app.next()
    app.expect(
        r"19\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
        with_eol=True,
    )
    # checking that we did not step in
    app.next()
    app.expect(r"in _ZN8__main__14kernel_sum_", with_eol=True)


def test_step(app: gdb):
    app.breakpoint("simple_dpex_func.py:19")
    app.run("simple_dpex_func.py")
    app.expect_hit_breakpoint(expected_location="simple_dpex_func.py:19")
    app.expect(
        r"19\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
        with_eol=True,
    )
    app.set_scheduler_lock()
    app.step()
    app.expect(r"func_sum.* at simple_dpex_func.py:12", with_eol=True)
    app.expect(r"12\s+result = a_in_func \+ b_in_func", with_eol=True)
    app.step()
    app.expect(
        r"13\s+return result",
        with_eol=True,
    )


@pytest.mark.parametrize("func", ["stepi", "nexti"])
def test_stepi(app: gdb, func: str):
    app.breakpoint("simple_dpex_func.py:19")
    app.run("simple_dpex_func.py")
    app.expect_hit_breakpoint(expected_location="simple_dpex_func.py:19")
    app.expect(
        r"19\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
        with_eol=True,
    )
    app.set_scheduler_lock()
    # Stepi/nexti steps over instruction, so the same source code line is
    # reached, but with a different instruction address.
    f = getattr(app, func)
    f()
    app.expect(
        r"19\s+c_in_kernel\[i\] = func_sum\(a_in_kernel\[i\], b_in_kernel\[i\]\)",
        with_eol=True,
    )
