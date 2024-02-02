# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Adds typeof implementations to Numba registry for all numba-dpex experimental
types.

"""

from numba.extending import typeof_impl

from numba_dpex.experimental.core.types.kernel_api.items import (
    ItemType,
    NdItemType,
)
from numba_dpex.kernel_api import AtomicRef, Item, NdItem

from .dpcpp_types import AtomicRefType


@typeof_impl.register(AtomicRef)
def typeof_atomic_ref(val: AtomicRef, ctx) -> AtomicRefType:
    """Returns a ``numba_dpex.experimental.dpctpp_types.AtomicRefType``
    instance for a Python AtomicRef object.

    Args:
        val (AtomicRef): Instance of the AtomicRef type.
        ctx : Numba typing context used for type inference.

    Returns: AtomicRefType object corresponding to the AtomicRef object.

    """
    dtype = typeof_impl(val.ref, ctx)

    return AtomicRefType(
        dtype=dtype,
        memory_order=val.memory_order.value,
        memory_scope=val.memory_scope.value,
        address_space=val.address_space.value,
    )


@typeof_impl.register(Item)
def typeof_item(val: Item, c):
    """Registers the type inference implementation function for a
    numba_dpex.kernel_api.Item PyObject.

    Args:
        val : An instance of numba_dpex.kernel_api.Item.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.experimental.core.types.kernel_api.items.ItemType
        instance.
    """
    return ItemType(val.ndim)


@typeof_impl.register(NdItem)
def typeof_nditem(val, c):
    """Registers the type inference implementation function for a
    numba_dpex.kernel_api.NdItem PyObject.

    Args:
        val : An instance of numba_dpex.kernel_api.NdItem.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.experimental.core.types.kernel_api.items.NdItemType
        instance.
    """
    return NdItemType(val.ndim)
