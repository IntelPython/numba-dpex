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
"""Common tools for testing debugging"""

import pathlib

import numba_dppy


def script_path(script):
    package_path = pathlib.Path(numba_dppy.__file__).parent
    return str(package_path / "examples/debug" / script)


def line_number(file_path, text):
    """Return line number of the text in the file"""
    with open(file_path, "r") as lines:
        for line_number, line in enumerate(lines):
            if text in line:
                return line_number + 1

    raise RuntimeError(f"Can not find {text} in {file_path}")


def breakpoint_by_mark(script, mark, offset=0):
    """Return breakpoint for the mark in the script

    Example: breakpoint_by_mark("script.py", "Set here") -> "script.py:25"
    """
    return f"{script}:{line_number(script_path(script), mark) + offset}"


def breakpoint_by_function(script, function):
    """Return breakpoint for the function in the script"""
    return breakpoint_by_mark(script, f"def {function}", 1)


def setup_breakpoint(
    app,
    breakpoint: str,
    script=None,
    expected_location=None,
    expected_line=None,
):
    if not script:
        script = breakpoint.split(" ")[0].split(":")[0]

    if not expected_location:
        expected_location = breakpoint.split(" ")[0]
        if not expected_location.split(":")[-1].isnumeric():
            expected_location = breakpoint_by_function(
                script, expected_location.split(":")[-1]
            )

    app.breakpoint(breakpoint)
    app.run(script)

    app.child.expect(fr"Thread .* hit Breakpoint .* at {expected_location}")

    if expected_line:
        app.child.expect(expected_line)
