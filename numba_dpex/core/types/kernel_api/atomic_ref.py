# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Collection of numba-dpex typing classes for kernel_api Python classes.
"""

from numba.core.types import Type


class AtomicRefType(Type):
    """numba-dpex internal type to represent a Python object of
    :class:`numba_dpex.kernel_api.AtomicRef`.
    """

    def __init__(
        self,
        dtype: Type,
        memory_order: int,
        memory_scope: int,
        address_space: int,
    ):
        """Creates an instance of the AtomicRefType representing a Python
        AtomicRef object.
        """
        self._dtype = dtype
        self._memory_order = memory_order
        self._memory_scope = memory_scope
        self._address_space = address_space
        name = (
            f"AtomicRef< {self._dtype}, "
            f"memory_order= {self._memory_order}, "
            f"memory_scope= {self._memory_scope}, "
            f"address_space= {self._address_space}>"
        )
        super().__init__(name=name)

    @property
    def memory_order(self) -> int:
        """Returns the integer value for a memory order that corresponds to
        kernel_api.MemoryOrder.
        """
        return self._memory_order

    @property
    def memory_scope(self) -> int:
        """Returns the integer value for a memory order that corresponds to
        kernel_api.MemoryScope.
        """
        return self._memory_scope

    @property
    def address_space(self) -> int:
        """Returns the integer value for a memory order that corresponds to
        kernel_api.AddressSpace.
        """
        return self._address_space

    @property
    def dtype(self):
        """Returns the Numba type of the object corresponding to the value
        stored in an AtomicRef.
        """
        return self._dtype

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
        )

    def cast_python_value(self, args):
        """The helper function is not overloaded and using it on the
        AtomicRefType throws a NotImplementedError.
        """
        raise NotImplementedError

    @property
    def mangling_args(self):
        args = [
            self.dtype,
            self.memory_order,
            self.memory_scope,
            self.address_space,
        ]
        return self.__class__.__name__, args
