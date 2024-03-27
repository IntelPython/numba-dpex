# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Imports and registers kernel_api_impl target-specific overloads.
"""


def init_kernel_api_spirv_overloads():
    """
    Imports the kernel_api.spirv overloads to make them available in numba-dpex.
    """
    from .kernel_api_impl.spirv.overloads import (
        _atomic_fence_overloads,
        _atomic_ref_overloads,
        _group_barrier_overloads,
        _index_space_id_overloads,
        _private_array_overloads,
    )
