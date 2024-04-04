# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Registers any custom lowering functions to default Numba lowering registry.
"""
from numba.core.imputils import Registry

from .types import KernelDispatcherType

registry = Registry()
lower_constant = registry.lower_constant


@lower_constant(KernelDispatcherType)
def dpex_dispatcher_const(context):
    """Dummy lowering function for a KernelDispatcherType object.

    The dummy lowering function for the KernelDispatcher types is added so that
    a :func:`numba_dpex.core.decorators.kernel` decorated function can be passed
    as an argument to dpjit.
    """
    return context.get_dummy_value()
