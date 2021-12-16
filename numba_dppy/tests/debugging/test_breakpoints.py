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

import pytest

from numba_dppy.tests._helper import skip_no_numba055, skip_no_gdb

pytestmark = skip_no_gdb


@skip_no_numba055
@pytest.mark.parametrize(
    "breakpoint, api",
    [
        ("side-by-side.py:25", "numba"),
        ("side-by-side.py:25", "numba-dppy-kernel"),
        ("common_loop_body_242", "numba"),
        ("common_loop_body", "numba-dppy-kernel"),
    ],
)
def test_breakpoint_with_condition_by_function_argument(app, breakpoint, api):
    """Function breakpoints and argument initializing

    Test that it is possible to set conditional breakpoint at the beginning
    of the function and use a function argument in the condition.

    Test for https://github.com/numba/numba/issues/7415
    SAT-4449
    """
    variable_name = "param_a"
    variable_value = "3"
    condition = f"{variable_name} == {variable_value}"

    app.breakpoint(f"{breakpoint} if {condition}")
    app.run(f"side-by-side.py --api={api}")

    app.child.expect(r"Thread .* hit Breakpoint .* at side-by-side.py:25")

    app.print(variable_name)

    app.child.expect(fr"\$1 = {variable_value}")
