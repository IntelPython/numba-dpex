# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a mock Python classes to represent ``sycl::item`` and
``sycl::nd_item``for prototyping numba_dpex kernel functions before they are JIT
compiled.
"""

from .ranges import Range


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
        if len(self._extent) == 1:
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

    @property
    def ndim(self) -> int:
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
    def __init__(self, global_item: Item, local_item: Item, group: any):
        # TODO: assert offset and dimensions
        self._global_item = global_item
        self._local_item = local_item
        self._group = group

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

    def get_group(self):
        """Returns the group.

        Returns:
            A group object."""
        return self._group

    @property
    def ndim(self) -> int:
        """Returns the rank of a NdItem object.

        Returns:
            int: Number of dimensions in the NdItem object
        """
        return self._global_item.ndim
