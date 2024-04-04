# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The kernel_api module provides a set of Python classes and functions that are
analogous to the C++ SYCL API. The kernel_api module is meant to allow
prototyping SYCL-like kernels in pure Python before compiling them using
numba_dpex.
"""

from .atomic_fence import atomic_fence
from .atomic_ref import AtomicRef
from .barrier import group_barrier
from .index_space_ids import Group, Item, NdItem
from .launcher import call_kernel
from .local_accessor import LocalAccessor
from .memory_enums import AddressSpace, MemoryOrder, MemoryScope
from .private_array import PrivateArray
from .ranges import NdRange, Range

__all__ = [
    "call_kernel",
    "group_barrier",
    "AddressSpace",
    "atomic_fence",
    "AtomicRef",
    "Group",
    "Item",
    "LocalAccessor",
    "MemoryOrder",
    "MemoryScope",
    "NdItem",
    "NdRange",
    "Range",
    "PrivateArray",
    "group_barrier",
    "call_kernel",
]
