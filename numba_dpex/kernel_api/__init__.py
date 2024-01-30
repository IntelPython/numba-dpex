# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The kernel_api module provides a set of Python classes and functions that are
analogous to the C++ SYCL API. The kernel_api module is meant to allow
prototyping SYCL-like kernels in pure Python before compiling them using
numba_dpex.
"""

from .atomic_ref import AtomicRef
from .barrier import group_barrier
from .index_space_ids import Item, NdItem
from .memory_enums import AddressSpace, MemoryOrder, MemoryScope
from .ranges import NdRange, Range

__all__ = [
    "AddressSpace",
    "AtomicRef",
    "MemoryOrder",
    "MemoryScope",
    "NdRange",
    "Range",
    "NdItem",
    "Item",
    "group_barrier",
]
