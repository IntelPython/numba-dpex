#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Common tools for testing debugging"""

import os
import pathlib


def script_path(script):
    current_path = os.getcwd()
    debug_script_path = os.path.join(current_path, "debug", script)
    return debug_script_path


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

    app.child.expect(rf"Thread .* hit Breakpoint .* at {expected_location}")

    if expected_line:
        app.child.expect(expected_line)
