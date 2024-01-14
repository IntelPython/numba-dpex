# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Defines constants that are used in other modules.

"""
from enum import Enum

__all__ = ["address_space", "calling_conv"]


class address_space:
    """The address space values supported by numba_dpex.

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
