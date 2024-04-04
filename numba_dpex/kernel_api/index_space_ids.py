# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a mock Python classes to represent ``sycl::item`` and
``sycl::nd_item``for prototyping numba_dpex kernel functions before they are JIT
compiled.
"""

from .ranges import Range


class Group:
    # pylint: disable=line-too-long
    """Analogue to the :sycl_group:`sycl::group <>` class.

    Represents a particular work-group within a parallel execution and
    provides API to extract various properties of the work-group. An instance
    of the class is not user-constructible. Users should use
    :func:`numba_dpex.kernel_api.NdItem.get_group` to access the Group to which
    a work-item belongs.
    """

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
        """Returns a specific coordinate of the multi-dimensional index of a group.

        Since the work-items in a work-group have a defined position within the
        global nd-range, the returned group id can be used along with the local
        id to uniquely identify the work-item in the global nd-range.

        Args:
            dim (int): An integral value between (1..3) for which the group
                index is returned.
        Returns:
            int: The coordinate for the ``dim`` dimension for the group's
            multi-dimensional index within an nd-range.
        Raises:
            ValueError: If the ``dim`` argument is not in the (1..3) interval.
        """
        if dim > len(self._index) - 1:
            raise ValueError(
                "Dimension value is out of bounds for the group index"
            )
        return self._index[dim]

    def get_group_linear_id(self):
        """Returns a linearized version of the work-group index.

        Returns:
            int: The linearized index for the group's position within an
            nd-range.
        """
        if self.dimensions == 1:
            return self.get_group_id(0)
        if self.dimensions == 2:
            return self.get_group_id(0) * self.get_group_range(
                1
            ) + self.get_group_id(1)
        return (
            (
                self.get_group_id(0)
                * self.get_group_range(1)
                * self.get_group_range(2)
            )
            + (self.get_group_id(1) * self.get_group_range(2))
            + (self.get_group_id(2))
        )

    def get_group_range(self, dim):
        """Returns the extent of the range of groups in an nd-range for given dimension.

        Args:
            dim (int): An integral value between (1..3) for which the group
                index is returned.
        Returns:
            int: The extent of group range for the specified dimension.
        """
        return self._group_range[dim]

    def get_group_linear_range(self):
        """Returns the total number of work-groups in the nd_range.

        Returns:
            int: Returns the number of groups in a parallel execution of an
            nd-range kernel.
        """
        num_wg = 1
        for i in range(self.dimensions):
            num_wg *= self.get_group_range(i)

        return num_wg

    def get_local_range(self, dim):
        """Returns the extent of the range of work-items in a work-group for given dimension.

        Args:
            dim (int): An integral value between (1..3) for which the group
                index is returned.
        Returns:
            int: The extent of the local work-item range for the specified
            dimension.
        """
        return self._local_range[dim]

    def get_local_linear_range(self):
        """Return the total number of work-items in the work-group.

        Returns:
            int: Returns the linearized size of the local range inside an
            nd-range.
        """
        num_wi = 1
        for i in range(self.dimensions):
            num_wi *= self.get_local_range(i)

        return num_wi

    @property
    def leader(self):
        """Return true if the caller work-item is the leader of the work-group.

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
        """Returns the dimensionality of the range to which the work-group belongs.

        Returns:
            int: Number of dimensions in the Group object
        """
        return self._global_range.ndim

    @leader.setter
    def leader(self, work_item_id):
        """Sets the leader attribute for the group."""
        self._leader = work_item_id


class Item:
    """Analogue to the :sycl_item:`sycl::item <>` class.

    Identifies the work-item in a parallel execution of a kernel launched with
    the :class:`.Range` index-space class.
    """

    def __init__(self, extent: Range, index: list):
        self._extent = extent
        self._index = index

    def get_linear_id(self):
        """Returns the linear id associated with this item for all dimensions.

        Returns:
            int: The linear id of the work item in the global range.
        """
        if self.dimensions == 1:
            return self.get_id(0)
        if self.dimensions == 2:
            return self.get_id(0) * self.get_range(1) + self.get_id(1)
        return (
            (self.get_id(0) * self.get_range(1) * self.get_range(2))
            + (self.get_id(1) * self.get_range(2))
            + (self.get_id(2))
        )

    def get_id(self, idx):
        """Get the id for a specific dimension.

        Returns:
            int: The id
        """
        return self._index[idx]

    def get_linear_range(self):
        """Return the total number of work-items in the work-group."""
        num_wi = 1
        for i in range(self.dimensions):
            num_wi *= self.get_range(i)

        return num_wi

    def get_range(self, idx):
        """Get the range size for a specific dimension.

        Returns:
            int: The size
        """
        return self._extent[idx]

    @property
    def dimensions(self) -> int:
        """Returns the number of dimensions of a Item object.

        Returns:
            int: Number of dimensions in the Item object
        """
        return self._extent.ndim


class NdItem:
    """Analogue to the :sycl_nditem:`sycl::nd_item <>` class.

    Identifies an instance of the function object executing at each point in an
    :class:`.NdRange`.
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
        """Get the linearized global id for the item for all dimensions.

        Returns:
            int: The global linear id.
        """
        # Instead of calling self._global_item.get_linear_id(), the linearization
        # logic is duplicated here so that the method can be JIT compiled by
        # numba-dpex and works in both Python and Numba nopython modes.
        if self.dimensions == 1:
            return self.get_global_id(0)
        if self.dimensions == 2:
            return self.get_global_id(0) * self.get_global_range(
                1
            ) + self.get_global_id(1)
        return (
            (
                self.get_global_id(0)
                * self.get_global_range(1)
                * self.get_global_range(2)
            )
            + (self.get_global_id(1) * self.get_global_range(2))
            + (self.get_global_id(2))
        )

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
        # Instead of calling self._local_item.get_linear_id(), the linearization
        # logic is duplicated here so that the method can be JIT compiled by
        # numba-dpex and works in both Python and Numba nopython modes.
        if self.dimensions == 1:
            return self.get_local_id(0)
        if self.dimensions == 2:
            return self.get_local_id(0) * self.get_local_range(
                1
            ) + self.get_local_id(1)
        return (
            (
                self.get_local_id(0)
                * self.get_local_range(1)
                * self.get_local_range(2)
            )
            + (self.get_local_id(1) * self.get_local_range(2))
            + (self.get_local_id(2))
        )

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

    def get_local_linear_range(self):
        """Return the total number of work-items in the work-group."""
        num_wi = 1
        for i in range(self.dimensions):
            num_wi *= self.get_local_range(i)

        return num_wi

    def get_global_linear_range(self):
        """Return the total number of work-items in the work-group."""
        num_wi = 1
        for i in range(self.dimensions):
            num_wi *= self.get_global_range(i)

        return num_wi

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
