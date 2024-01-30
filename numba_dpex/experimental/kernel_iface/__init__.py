# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The kernel_iface provides a set of Python classes and functions that are
analogous to the C++ SYCL API. The kernel_iface API is meant to allow
prototyping SYCL-like kernels in pure Python before compiling them using
numba_dpex.kernel.
"""

from .atomic_ref import AtomicRef
from .barrier import group_barrier, sub_group_barrier
from .memory_enums import AddressSpace, MemoryOrder, MemoryScope

__all__ = [
    "group_barrier",
    "sub_group_barrier",
    "AddressSpace",
    "AtomicRef",
    "MemoryOrder",
    "MemoryScope",
]
