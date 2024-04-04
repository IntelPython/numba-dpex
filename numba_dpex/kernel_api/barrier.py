# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Python functions that simulate SYCL's group_barrier function.
"""

from .index_space_ids import Group
from .memory_enums import MemoryScope


def group_barrier(
    group: Group, fence_scope: MemoryScope = MemoryScope.WORK_GROUP
):
    """Performs a barrier operation across all work-items in a work-group.

    The function is equivalent to the ``sycl::group_barrier`` function. It
    synchronizes work within a group of work-items. All the work-items
    of the group must execute the barrier call before any work-item
    continues execution beyond the barrier.

    The ``group_barrier`` performs a memory fence operation ensuring that memory
    accesses issued before the barrier are not re-ordered with those issued
    after the barrier. All work-items in group G execute a release fence prior
    to synchronizing at the barrier, all work-items in group G execute an
    acquire fence afterwards, and there is an implicit synchronization of these
    fences as if provided by an explicit atomic operation on an atomic object.

    .. important::
        The function is not implemented yet for pure CPython execution and is
        only supported in JIT compiled mode of execution.

    Args:
        group (Group): Indicates the work-group inside which the barrier is to
            be executed.
        fence_scope (MemoryScope) (optional): scope of any memory
            consistency operations that are performed by the barrier.
    Raises:
        NotImplementedError: When the function is called directly from Python.
    """

    # TODO: A pure Python simulation of a group_barrier will be added later.
    raise NotImplementedError
