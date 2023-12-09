# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Contains experimental features that are meant as engineering preview and not
yet production ready.
"""

from numba.core.imputils import Registry

from .decorators import device_func, kernel
from .kernel_dispatcher import KernelDispatcher
from .launcher import call_kernel, call_kernel_async
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
    "device_func",
    "kernel",
    "call_kernel",
    "call_kernel_async",
    "IntEnumLiteral",
    "KernelDispatcher",
]
