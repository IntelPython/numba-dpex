# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable


class Range(tuple):
    """A data structure to encapsulate a single kernel launch parameter.

    The range is an abstraction that describes the number of elements
    in each dimension of buffers and index spaces. It can contain
    1, 2, or 3 numbers, depending on the dimensionality of the
    object it describes.

    This is just a wrapper class on top of a 3-tuple. The kernel launch
    parameter is consisted of three int's. This class basically mimics
    the behavior of `sycl::range`.
    """

    def __new__(cls, dim0, dim1=None, dim2=None):
        """Constructs a 1, 2, or 3 dimensional range.

        Args:
            dim0 (int): The range of the first dimension.
            dim1 (int, optional): The range of second dimension.
                                    Defaults to None.
            dim2 (int, optional): The range of the third dimension.
                                    Defaults to None.

        Raises:
            TypeError: If dim0 is not an int.
            TypeError: If dim1 is not an int.
            TypeError: If dim2 is not an int.
        """
        if not isinstance(dim0, int):
            raise TypeError("dim0 of a Range must be an int.")
        _values = [dim0]
        if dim1:
            if not isinstance(dim1, int):
                raise TypeError("dim1 of a Range must be an int.")
            _values.append(dim1)
            if dim2:
                if not isinstance(dim2, int):
                    raise TypeError("dim2 of a Range must be an int.")
                _values.append(dim2)
        return super(Range, cls).__new__(cls, tuple(_values))

    def get(self, index):
        """Returns the range of a single dimension.

        Args:
            index (int): The index of the dimension, i.e. [0,2]

        Returns:
            int: The range of the dimension indexed by `index`.
        """
        return self[index]

    def size(self):
        """Returns the size of a range.

        Returns the size of a range by multiplying
        the range of the individual dimensions.

        Returns:
            int: The size of a range.
        """
        n = len(self)
        if n > 2:
            return self[0] * self[1] * self[2]
        elif n > 1:
            return self[0] * self[1]
        else:
            return self[0]


class NdRange:
    """A class to encapsulate all kernel launch parameters.

    The NdRange defines the index space for a work group as well as
    the global index space. It is passed to parallel_for to execute
    a kernel on a set of work items.

    This class basically contains two Range object, one for the global_range
    and the other for the local_range. The global_range parameter contains
    the global index space and the local_range parameter contains the index
    space of a work group. This class mimics the behavior of `sycl::nd_range`
    class.
    """

    def __init__(self, global_size, local_size):
        """Constructor for NdRange class.

        Args:
            global_size (Range or tuple of int's): The values for
                the global_range.
            local_size (Range or tuple of int's, optional): The values for
                the local_range. Defaults to None.
        """
        if isinstance(global_size, Range):
            self._global_range = global_size
        elif isinstance(global_size, Iterable):
            self._global_range = Range(*global_size)
        else:
            raise TypeError(
                "Unknown argument type for NdRange global_size, "
                + "must be of either type Range or Iterable of int's."
            )

        if isinstance(local_size, Range):
            self._local_range = local_size
        elif isinstance(local_size, Iterable):
            self._local_range = Range(*local_size)
        else:
            raise TypeError(
                "Unknown argument type for NdRange local_size, "
                + "must be of either type Range or Iterable of int's."
            )

    @property
    def global_range(self):
        """Accessor for global_range.

        Returns:
            Range: The `global_range` `Range` object.
        """
        return self._global_range

    @property
    def local_range(self):
        """Accessor for local_range.

        Returns:
            Range: The `local_range` `Range` object.
        """
        return self._local_range

    def get_global_range(self):
        """Returns a Range defining the index space.

        Returns:
            Range: A `Range` object defining the index space.
        """
        return self._global_range

    def get_local_range(self):
        """Returns a Range defining the index space of a work group.

        Returns:
            Range: A `Range` object to specify index space of a work group.
        """
        return self._local_range

    def __str__(self):
        """str() function for NdRange class.

        Returns:
            str: str representation for NdRange class.
        """
        return (
            "(" + str(self._global_range) + ", " + str(self._local_range) + ")"
        )

    def __repr__(self):
        """repr() function for NdRange class.

        Returns:
            str: str representation for NdRange class.
        """
        return self.__str__()
