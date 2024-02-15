# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Adds typeof implementations to Numba registry for all numba-dpex experimental
types.

"""

from numba.extending import typeof_impl

from numba_dpex.kernel_api import AtomicRef

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
