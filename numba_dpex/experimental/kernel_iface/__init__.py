# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""The kernel_iface provides a set of Python classes and functions that are
analogous to the C++ SYCL API. The kernel_iface API is meant to allow
prototyping SYCL-like kernels in pure Python before compiling them using
numba_dpex.kernel.
"""

from .memory_enums import AddressSpace, MemoryOrder, MemoryScope

__all__ = ["AddressSpace", "MemoryOrder", "MemoryScope"]
