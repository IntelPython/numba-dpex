# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

from numba.core import cgutils, errors, types


class RangeType(types.Type):
    """Numba-dpex type corresponding to
    :class:`numba_dpex.kernel_api.ranges.Range`
    """

    def __init__(self, ndim: int):
        self._ndim = ndim
        if ndim < 1 or ndim > 3:
            raise errors.TypingError(
                "RangeType can only have 1,2, or 3 dimensions"
            )
        super(RangeType, self).__init__(name="Range<" + str(ndim) + ">")

    @property
    def ndim(self):
        return self._ndim

    @property
    def key(self):
        return self._ndim

    @property
    def mangling_args(self):
        args = [self.ndim]
        return self.__class__.__name__, args


class NdRangeType(types.Type):
    """Numba-dpex type corresponding to
    :class:`numba_dpex.kernel_api.ranges.NdRange`
    """

    def __init__(self, ndim: int):
        self._ndim = ndim
        if ndim < 1 or ndim > 3:
            raise errors.TypingError(
                "RangeType can only have 1,2, or 3 dimensions"
            )
        super(NdRangeType, self).__init__(name="NdRange<" + str(ndim) + ">")

    @property
    def ndim(self):
        return self._ndim

    @property
    def key(self):
        return self._ndim

    @property
    def mangling_args(self):
        args = [self.ndim]
        return self.__class__.__name__, args
