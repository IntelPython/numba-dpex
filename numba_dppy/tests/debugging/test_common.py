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
"""Tests for common tools"""

import pytest

from .common import breakpoint_by_function, breakpoint_by_mark


@pytest.mark.parametrize(
    "file_name, mark, expected",
    [("side-by-side.py", "Set breakpoint here", "side-by-side.py:25")],
)
def test_breakpoint_by_mark(file_name, mark, expected):
    assert expected == breakpoint_by_mark(file_name, mark)


@pytest.mark.parametrize(
    "file_name, function, expected",
    [("side-by-side.py", "common_loop_body", "side-by-side.py:25")],
)
def test_breakpoint_by_function(file_name, function, expected):
    assert expected == breakpoint_by_function(file_name, function)
