#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for side-by-side.py script"""


import pytest

from numba_dpex.tests._helper import skip_no_gdb

pytestmark = skip_no_gdb


@pytest.mark.parametrize("api", ["numba", "numba-ndpx-kernel"])
def test_breakpoint_row_number(app, api):
    """Test for checking numba and numba-dpex debugging side-by-side."""

    app.breakpoint("side-by-side.py:16")
    app.run("side-by-side.py --api={api}".format(api=api))

    app.child.expect(r"Breakpoint .* at side-by-side.py:16")
    app.child.expect(r"16\s+param_c = param_a \+ 10")
