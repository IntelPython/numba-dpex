# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
