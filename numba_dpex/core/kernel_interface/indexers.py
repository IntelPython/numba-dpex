# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


class Range:
    """Defines an 1, 2, or 3 dimensional index space over which a kernel is
    executed.

    The Range class is analogous to SYCL's ``sycl::range`` class.
    """

    def __init__(self, dim0, dim1=None, dim2=None):
        self._dim0 = dim0
        self._dim1 = dim1
        self._dim2 = dim2

        if not self._dim0:
            raise ValueError("Outermost dimension of a Range cannot be None.")

        if self._dim2 and not self._dim1:
            raise ValueError(
                "A 3rd dimension cannot be specified if a 2nd dimension "
                "was not specified."
            )

        if not isinstance(self._dim0, int):
            raise ValueError(
                "The size of a dimension needs to be specified as an "
                "integer value."
            )

        if self._dim1 and not isinstance(self._dim1, int):
            raise ValueError(
                "The size of a dimension needs to be specified as an "
                "integer value."
            )

        if self._dim2 and not isinstance(self._dim2, int):
            raise ValueError(
                "The size of a dimension needs to be specified as an "
                "integer value."
            )

    def get(self, dim):
        """Returns the size of the Range in a given dimension."""
        if not isinstance(dim, int):
            raise ValueError(
                "The dimension needs to be specified as an integer value."
            )

        if dim == 0:
            return self._dim0
        elif dim == 1:
            return self._dim1
        elif dim == 2:
            return self._dim2
        else:
            raise ValueError(
                "Unsupported dimension number. A Range "
                "only has 1, 2, or 3 dimensions."
            )

    @property
    def size(self):
        """Returns cummulative size of the Range."""
        size = self._dim0
        if self._dim1:
            size *= self._dim1
        if self._dim2:
            size *= self._dim2

        return size

    @property
    def rank(self):
        """Returns the rank (dimensionality) of the Range."""
        rank = 1

        # We already checked in init that if dim2 is set that dim1 has
        # to be set as well
        if self._dim1:
            rank += 1
        elif self._dim2:
            rank += 1

        return rank


class NdRange:
    """Defines the iteration domain of both the work-groups and the overall
    dispatch.

    The nd_range comprises two ranges: the whole range over which the kernel is
    to be executed (global_size), and the range of each work group (local_size).
    """

    def _check_ndrange(self):
        """Checks if the specified nd_range (global_range, local_range) are
        valid.
        """
        if len(self._local_range) != len(self._global_range):
            raise ValueError(
                "Global and local ranges should have same number of dimensions."
            )

        for i in range(len(self._global_range)):
            if self._global_range[i] % self._local_range[i] != 0:
                raise ValueError(
                    "The global work groups must be evenly divisible by the"
                    " local work items evenly."
                )

    def _set_range(self, range):
        normalized_range = None
        if isinstance(range, int):
            normalized_range = Range(range)
        elif isinstance(range, tuple) or isinstance(range, list):
            if len(range) == 1:
                normalized_range = Range(dim0=range[0])
            elif len(range == 2):
                normalized_range = Range(dim0=range[0], dim1=range[1])
            elif len(range == 3):
                normalized_range = Range(
                    dim0=range[0],
                    dim1=range[1],
                    dim2=range[2],
                )
            else:
                raise ValueError(
                    "A Range cannot have more than three dimensions."
                )
        return normalized_range

    def __init__(self, *, global_range, local_range) -> None:
        if global_range is None:
            raise ValueError("Global range cannot be None.")
        if local_range is None:
            raise ValueError("Local range cannot be None.")

        self._global_range = self._set_range(global_range)
        self._local_range = self._set_range(local_range)

        # check if the ndrange is sane
        self._check_ndrange()

    @property
    def global_range(self):
        """Return the constituent global range."""
        return self._global_size

    @property
    def local_range(self):
        """Return the constituent local range."""
        return self._local_size
