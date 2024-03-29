# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Python functions that simulate SYCL's atomic_fence primitives.
"""
from .memory_enums import MemoryOrder, MemoryScope


def atomic_fence(
    memory_order: MemoryOrder, memory_scope: MemoryScope
):  # pylint: disable=unused-argument
    """Performs a memory fence operations across all work-items.

    The function is equivalent to the ``sycl::atomic_fence`` function and
    controls the order of memory accesses (loads and stores) by individual
    work-items.

    .. important::
        The function is a no-op during CPython execution and only available in
        JIT compiled mode of execution.

    Args:
        memory_order (MemoryOrder): The memory synchronization order.
        memory_scope (MemoryScope): The set of work-items and devices to which
            the memory ordering constraints apply.

    """
