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

import os
import os.path


def atomic_support_present():
    if os.path.isfile(os.path.join(os.path.dirname(__file__), "atomic_ops.spir")):
        return True
    else:
        return False


def get_atomic_spirv_path():
    if atomic_support_present():
        return os.path.join(os.path.dirname(__file__), "atomic_ops.spir")
    else:
        return None


def read_atomic_spirv_file():
    path = get_atomic_spirv_path()
    if path:
        with open(path, "rb") as fin:
            spirv = fin.read()
        return spirv
    else:
        return None
