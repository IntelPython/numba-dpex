# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Implements a mock Python class to represent ``sycl::atomic_ref`` for
prototyping numba_dpex kernel functions before they are JIT compiled.
"""

from .memory_enums import AddressSpace, MemoryOrder, MemoryScope


class AtomicRef:
    """Analogue to the ``sycl::atomic_ref`` type. An atomic reference is a
    view into a data container that can be then updated atomically using any of
    the ``fetch_*`` member functions of the class.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        ref,
        index=0,
        memory_order=MemoryOrder.RELAXED,
        memory_scope=MemoryScope.DEVICE,
        address_space=AddressSpace.GLOBAL,
    ):
        """A Python stub to represent a SYCL AtomicRef class. An AtomicRef
        object represents a single element view into a Python array-like object
        that can be then updated using various fetch_* type functions.

        To maintain analogy with SYCL's API, the AtomicRef constructor takes
        optional ``memory_order``, ``memory_scope``, and ``address_space``
        arguments that are ignored in Python execution.
        """
        self._memory_order = memory_order
        self._memory_scope = memory_scope
        self._address_space = address_space
        self._ref = ref
        self._index = index

    @property
    def ref(self):
        """Returns the value stored in the AtomicRef._ref[_index]."""
        return self._ref[self._index]

    @property
    def memory_order(self) -> MemoryOrder:
        """Returns the MemoryOrder value used to define the AtomicRef."""
        return self._memory_order

    @property
    def memory_scope(self) -> MemoryScope:
        """Returns the MemoryScope value used to define the AtomicRef."""
        return self._memory_scope

    @property
    def address_space(self) -> AddressSpace:
        """Returns the AddressSpace value used to define the AtomicRef."""
        return self._address_space

    def fetch_add(self, val):
        """Adds the operand ``val`` to the object referenced by the AtomicRef
        and assigns the result to the value of the referenced object. Returns
        the original value of the object.

        Args:
            val : Value to be added to the object referenced by the AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] += val
        return old

    def fetch_sub(self, val):
        """Subtracts the operand ``val`` to the object referenced by the
        AtomicRef and assigns the result to the value of the referenced object.
        Returns the original value of the object.

        Args:
            val : Value to be subtracted to the object referenced by the
            AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] -= val
        return old

    def fetch_min(self, val):
        """Calculates the minimum value of the operand ``val`` and the object
        referenced by the AtomicRef and assigns the result to the value of the
        referenced object. Returns the original value of the object.

        Args:
            val : Value to be compared against to the object referenced by the
            AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] = min(old, val)
        return old

    def fetch_max(self, val):
        """Calculates the maximum value of the operand ``val`` and the object
        referenced by the AtomicRef and assigns the result to the value of the
        referenced object. Returns the original value of the object.

        Args:
            val : Value to be compared against to the object referenced by the
            AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] = max(old, val)
        return old

    def fetch_and(self, val):
        """Calculates the bitwise AND of the operand ``val`` and the object
        referenced by the AtomicRef and assigns the result to the value of the
        referenced object. Returns the original value of the object.

        Args:
            val : Value to be bitwise ANDed against to the object referenced by
            the AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] &= val
        return old

    def fetch_or(self, val):
        """Calculates the bitwise OR of the operand ``val`` and the object
        referenced by the AtomicRef and assigns the result to the value of the
        referenced object. Returns the original value of the object.

        Args:
            val : Value to be bitwise ORed against to the object referenced by
            the AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] |= val
        return old

    def fetch_xor(self, val):
        """Calculates the bitwise XOR of the operand ``val`` and the object
        referenced by the AtomicRef and assigns the result to the value of the
        referenced object. Returns the original value of the object.

        Args:
            val : Value to be bitwise XORed against to the object referenced by
            the AtomicRef.

        Returns: The original value of the object referenced by the AtomicRef.

        """
        old = self._ref[self._index]
        self._ref[self._index] ^= val
        return old
