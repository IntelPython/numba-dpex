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

__all__ = [
    "SPIRVKernelDispatcher",
]
