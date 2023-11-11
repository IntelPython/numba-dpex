# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.types import Type


class AtomicRefType(Type):
    """numba-dpex internal type to represent an object of
    :class:`numba_dpex.core.kernel_interface.dpcpp_iface.AtomicRef`.
    """

    def __init__(
        self,
        dtype,
        memory_order,
        memory_scope,
        address_space,
        has_aspect_atomic64,
    ):
        super(AtomicRefType, self).__init__(name="AtomicRef")
        self._dtype = dtype
        self._memory_order = memory_order
        self._memory_scope = memory_scope
        self._address_space = address_space
        self._has_aspect_atomic64 = has_aspect_atomic64

    @property
    def memory_order(self):
        return self._memory_order

    @property
    def memory_scope(self):
        return self._memory_scope

    @property
    def address_space(self):
        return self._address_space

    @property
    def dtype(self):
        return self._dtype

    @property
    def has_aspect_atomic64(self):
        return self._has_aspect_atomic64

    @property
    def key(self):
        """
        A property used for __eq__, __ne__ and __hash__.
        """
        return (
            self.dtype,
            self.memory_order,
            self.memory_scope,
            self.address_space,
            self.has_aspect_atomic64,
        )
