# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Python functions that simulate SYCL's atomic_fence primitives.
"""


def atomic_fence(memory_order, memory_scope):  # pylint: disable=unused-argument
    """The function for performing memory fence across all work-items.
    Modeled after ``sycl::atomic_fence`` function.
    It provides control over re-ordering of memory load
    and store operations. The ``atomic_fence`` function acts as a
    fence across all work-items and devices specified by a
    memory_scope argument.

    Args:
    memory_order: The memory synchronization order.

    memory_scope: The set of work-items and devices to which
    the memory ordering constraints apply.

    """
