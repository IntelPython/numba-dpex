# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Contains experimental features that are meant as engineering preview and not
yet production ready.
"""

from numba.core.imputils import Registry

from .decorators import kernel
from .dpcpp_iface import (
    AddressSpace,
    AtomicRef,
    MemoryOrder,
    MemoryScope,
    atomic_fence,
    group_barrier,
    sub_group_barrier,
)
from .dpcpp_types import AtomicRefType
from .kernel_dispatcher import KernelDispatcher
from .launcher import call_kernel
from .literal_intenum_type import IntEnumLiteral
from .models import *
from .types import KernelDispatcherType

registry = Registry()
lower_constant = registry.lower_constant


@lower_constant(KernelDispatcherType)
def dpex_dispatcher_const(context):
    """Dummy lowerer for a KernelDispatcherType object. It is added so that a
    KernelDispatcher can be passed as an argument to dpjit.
    """
    return context.get_dummy_value()


__all__ = [
    "atomic_fence",
    "group_barrier",
    "sub_group_barrier",
    "kernel",
    "call_kernel",
    "AddressSpace",
    "AtomicRef",
    "AtomicRefType",
    "IntEnumLiteral",
    "KernelDispatcher",
    "MemoryOrder",
    "MemoryScope",
]
