# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


class Range:
    """Defines an 1,2,or 3 dimensional index space over which a kernel is
    executed.

    The Range class is analogous to SYCL's ``sycl::range`` class.
    """

    def __init__(self, dim0, dim1=None, dim2=None):
        self._dim0 = dim0
        self._dim1 = dim1
        self._dim2 = dim2

        if not self._dim0:
            raise ValueError

        if self._dim2 and not self._dim1:
            raise ValueError

        if not isinstance(self._dim0, int):
            raise ValueError

        if self._dim1 and not isinstance(self._dim1, int):
            raise ValueError

        if self._dim2 and not isinstance(self._dim2, int):
            raise ValueError

    def get(self, dim):
        if not isinstance(dim, int):
            raise ValueError

        if dim == 0:
            return self._dim0
        elif dim == 1:
            return self._dim1
        elif dim == 2:
            return self._dim3
        else:
            raise ValueError

    def size(self):
        size = self._dim0
        if self._dim1:
            size *= self._dim1
        if self._dim2:
            size *= self._dim2

        return size

    def rank(self):
        rank = 1

        # We already checked in init that if dim2 is set that dim1 has
        # to be set as well
        if self._dim1:
            rank += 1
        elif self._dim2:
            rank += 1

        return rank
