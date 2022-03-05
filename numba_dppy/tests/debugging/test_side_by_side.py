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
"""Tests for side-by-side.py script"""


import pytest

from numba_dppy.tests._helper import skip_no_gdb

pytestmark = skip_no_gdb


@pytest.mark.parametrize("api", ["numba", "numba-dpex-kernel"])
def test_breakpoint_row_number(app, api):
    """Test for checking numba and numba-dppy debugging side-by-side."""

    app.breakpoint("side-by-side.py:25")
    app.run("side-by-side.py --api={api}".format(api=api))

    app.child.expect(r"Breakpoint .* at side-by-side.py:25")
    app.child.expect(r"25\s+param_c = param_a \+ 10")
