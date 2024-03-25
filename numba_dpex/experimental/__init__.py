# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Contains experimental features that are meant as engineering preview and not
yet production ready.
"""


# Temporary so that Range and NdRange work in experimental call_kernel
from numba_dpex.core.boxing import *
from numba_dpex.kernel_api_impl.spirv.dispatcher import SPIRVKernelDispatcher

from . import typeof
from ._kernel_dpcpp_spirv_overloads import (
    _atomic_fence_overloads,
    _atomic_ref_overloads,
    _group_barrier_overloads,
    _index_space_id_overloads,
    _private_array_overloads,
)

__all__ = [
    "SPIRVKernelDispatcher",
]
