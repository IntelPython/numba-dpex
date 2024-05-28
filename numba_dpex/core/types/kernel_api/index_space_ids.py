# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Defines numba types for Item and NdItem classes"""

from numba.core import errors, types


class GroupType(types.Type):
    """Numba-dpex type corresponding to :class:`numba_dpex.kernel_api.Group`"""

    def __init__(self, ndim: int):
        self._ndim = ndim
        if ndim < 1 or ndim > 3:
            raise errors.TypingError(
                "ItemType can only have 1, 2 or 3 dimensions"
            )
        super().__init__(name="Group<" + str(ndim) + ">")

    @property
    def ndim(self):
        """Returns number of dimensions"""
        return self._ndim

    @property
    def key(self):
        """Numba type specific overload"""
        return self._ndim

    def cast_python_value(self, args):
        raise NotImplementedError

    @property
    def mangling_args(self):
        args = [self.ndim]
        return self.__class__.__name__, args


class ItemType(types.Type):
    """Numba-dpex type corresponding to :class:`numba_dpex.kernel_api.Item`"""

    def __init__(self, ndim: int):
        self._ndim = ndim
        if ndim < 1 or ndim > 3:
            raise errors.TypingError(
                "ItemType can only have 1, 2 or 3 dimensions"
            )
        super().__init__(name="Item<" + str(ndim) + ">")

    @property
    def ndim(self):
        """Returns number of dimensions"""
        return self._ndim

    @property
    def key(self):
        """Numba type specific overload"""
        return self._ndim

    @property
    def mangling_args(self):
        args = [self.ndim]
        return self.__class__.__name__, args

    def cast_python_value(self, args):
        raise NotImplementedError


class NdItemType(types.Type):
    """Numba-dpex type corresponding to :class:`numba_dpex.kernel_api.NdItem`"""

    def __init__(self, ndim: int):
        self._ndim = ndim
        if ndim < 1 or ndim > 3:
            raise errors.TypingError(
                "ItemType can only have 1, 2 or 3 dimensions"
            )
        super().__init__(name="NdItem<" + str(ndim) + ">")

    @property
    def ndim(self):
        """Returns number of dimensions"""
        return self._ndim

    @property
    def key(self):
        """Numba type specific overload"""
        return self._ndim

    @property
    def mangling_args(self):
        args = [self.ndim]
        return self.__class__.__name__, args

    def cast_python_value(self, args):
        raise NotImplementedError
