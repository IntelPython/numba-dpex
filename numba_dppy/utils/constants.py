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

"""Defines constants that are used in other modules.

"""
from enum import Enum

__all__ = ["address_space", "calling_conv"]


class address_space:
    """The address space values supported by numba-dppy.

    ==================   ============
    Address space        Value
    ==================   ============
    PRIVATE              0
    GLOBAL               1
    CONSTANT             2
    LOCAL                3
    GENERIC              4
    ==================   ============

    """

    PRIVATE = 0
    GLOBAL = 1
    CONSTANT = 2
    LOCAL = 3
    GENERIC = 4


class calling_conv:
    CC_SPIR_KERNEL = "spir_kernel"
    CC_SPIR_FUNC = "spir_func"
