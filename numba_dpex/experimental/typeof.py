# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.extending import typeof_impl

from .dpcpp_iface import AtomicRef
from .dpcpp_types import AtomicRefType


@typeof_impl.register(AtomicRef)
def typeof_atomic_ref(val, c):
    return AtomicRefType(val)
