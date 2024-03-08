# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements the SPIR-V overloads for the kernel_api.PrivateArray class.
"""

from numba.core.imputils import Registry

registry = Registry()
lower = registry.lower
