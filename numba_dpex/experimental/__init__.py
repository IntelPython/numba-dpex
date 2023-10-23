# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.imputils import Registry

from .decorators import kernel
from .kernel_dispatcher import KernelDispatcher
from .launcher import call_kernel
from .models import *
from .types import KernelDispatcherType

registry = Registry()
lower_constant = registry.lower_constant


@lower_constant(KernelDispatcherType)
def dpex_dispatcher_const(context, builder, ty, pyval):
    return context.get_dummy_value()


__all__ = ["kernel", "KernelDispatcher", "call_kernel"]
