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
from .memory_enums import AddressSpace, MemoryOrder, MemoryScope

__all__ = ["AddressSpace", "AtomicRef", "MemoryOrder", "MemoryScope"]
