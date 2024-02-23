# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a mock Python classes to represent ``sycl::item`` and
``sycl::nd_item``for prototyping numba_dpex kernel functions before they are JIT
compiled.
"""

from .ranges import Range


class Group:
    """Analogue to the ``sycl::group`` type."""

    def __init__(
        self,
        global_range: Range,
        local_range: Range,
        group_range: Range,
        index: list,
    ):
        self._global_range = global_range
        self._local_range = local_range
        self._group_range = group_range
        self._index = index
        self._leader = False

    def get_group_id(self, dim):
        """Returns the index of the work-group within the global nd-range for
        specified dimension.

        Since the work-items in a work-group have a defined position within the
        global nd-range, the returned group id can be used along with the local
        id to uniquely identify the work-item in the global nd-range.
        """
        if dim > len(self._index) - 1:
            raise ValueError(
                "Dimension value is out of bounds for the group index"
            )
        return self._index[dim]

    def get_group_linear_id(self):
        """Returns a linearized version of the work-group index."""
        if len(self._index) == 1:
            return self._index[0]
        if len(self._index) == 2:
            return self._index[0] * self._group_range[1] + self._index[1]
        return (
            (self._index[0] * self._group_range[1] * self._group_range[2])
            + (self._index[1] * self._group_range[2])
            + (self._index[2])
        )

    def get_group_range(self, dim):
        """Returns a the extent of the range representing the number of groups
        in the nd-range for a specified dimension.
        """
        return self._group_range[dim]

    def get_group_linear_range(self):
        """Return the total number of work-groups in the nd_range."""
        num_wg = 1
        for ext in self._group_range:
            num_wg *= ext

        return num_wg

    def get_local_range(self, dim):
        """Returns the extent of the SYCL range representing all dimensions
        of the local range for a specified dimension. This local range may
        have been provided by the programmer, or chosen by the SYCL runtime.
        """
        return self._local_range[dim]

    def get_local_linear_range(self):
        """Return the total number of work-items in the work-group."""
        num_wi = 1
        for ext in self._local_range:
            num_wi *= ext

        return num_wi

    @property
    def leader(self):
        """Return true for exactly one work-item in the work-group, if the
        calling work-item is the leader of the work-group, and false for all
        other work-items in the work-group.

        The leader of the work-group is determined during construction of the
        work-group, and is invariant for the lifetime of the work-group. The
        leader of the work-group is guaranteed to be the work-item with a
        local id of 0.


        Returns:
            bool: If the work item is the designated leader of the
        """
        return self._leader

    @property
    def dimensions(self) -> int:
        """Returns the rank of a Group object.
        Returns:
            int: Number of dimensions in the Group object
        """
        return self._global_range.ndim

    @leader.setter
    def leader(self, work_item_id):
        """Sets the leader attribute for the group."""
        self._leader = work_item_id


class Item:
    """Analogue to the ``sycl::item`` type. Identifies an instance of the
    function object executing at each point in an Range.
    """

    def __init__(self, extent: Range, index: list):
        self._extent = extent
        self._index = index

    def get_linear_id(self):
        """Get the linear id associated with this item for all dimensions.
        Original implementation could be found at ``sycl::item_base`` class.

        Returns:
            int: The linear id.
        """
        if len(self._extent) == 1:
            return self._index[0]
        if len(self._extent) == 2:
            return self._index[0] * self._extent[1] + self._index[1]
        return (
            (self._index[0] * self._extent[1] * self._extent[2])
            + (self._index[1] * self._extent[2])
            + (self._index[2])
        )

    def get_id(self, idx):
        """Get the id for a specific dimension.

        Returns:
            int: The id
        """
        return self._index[idx]

    def get_range(self, idx):
        """Get the range size for a specific dimension.

        Returns:
            int: The size
        """
        return self._extent[idx]

    @property
    def dimensions(self) -> int:
        """Returns the rank of a Item object.

        Returns:
            int: Number of dimensions in the Item object
        """
        return self._extent.ndim


class NdItem:
    """Analogue to the ``sycl::nd_item`` type. Identifies an instance of the
    function object executing at each point in an NdRange.
    """

    # TODO: define group type
    def __init__(self, global_item: Item, local_item: Item, group: Group):
        # TODO: assert offset and dimensions
        self._global_item = global_item
        self._local_item = local_item
        self._group = group
        if self.get_local_linear_id() == 0:
            self._group.leader = True

    def get_global_id(self, idx):
        """Get the global id for a specific dimension.

        Returns:
            int: The global id
        """
        return self._global_item.get_id(idx)

    def get_global_linear_id(self):
        """Get the global linear id associated with this item for all
        dimensions.

        Returns:
            int: The global linear id.
        """
        return self._global_item.get_linear_id()

    def get_local_id(self, idx):
        """Get the local id for a specific dimension.

        Returns:
            int: The local id
        """
        return self._local_item.get_id(idx)

    def get_local_linear_id(self):
        """Get the local linear id associated with this item for all
        dimensions.

        Returns:
            int: The local linear id.
        """
        return self._local_item.get_linear_id()

    def get_global_range(self, idx):
        """Get the global range size for a specific dimension.

        Returns:
            int: The size
        """
        return self._global_item.get_range(idx)

    def get_local_range(self, idx):
        """Get the local range size for a specific dimension.

        Returns:
            int: The size
        """
        return self._local_item.get_range(idx)

    def get_group(self):
        """Returns the group.

        Returns:
            A group object."""
        return self._group

    @property
    def dimensions(self) -> int:
        """Returns the rank of a NdItem object.

        Returns:
            int: Number of dimensions in the NdItem object
        """
        return self._global_item.dimensions
