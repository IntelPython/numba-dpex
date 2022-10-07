# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import string


def wrapper_function(args, code, globals, function_name="func"):
    function_text = f"""\
def {function_name}({args}):
    return {code}
"""
    return compile_function(function_text, function_name, globals)


def args_string(args_count):
    return ", ".join(list(string.ascii_lowercase[:args_count]))


def compile_function(function_text, function_name, globals):
    locals = {}
    exec(function_text, globals, locals)
    return locals[function_name]
