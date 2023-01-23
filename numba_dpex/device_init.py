# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# Re export
from .ocl.stubs import (
    GLOBAL_MEM_FENCE,
    LOCAL_MEM_FENCE,
    atomic,
    barrier,
    get_global_id,
    get_global_size,
    get_group_id,
    get_local_id,
    get_local_size,
    get_num_groups,
    get_work_dim,
    local,
    mem_fence,
    private,
    sub_group_barrier,
)

"""
We are importing dpnp stub module to make Numba recognize the
module when we rename Numpy functions.
"""
from .dpnp_iface.stubs import dpnp

DEFAULT_LOCAL_SIZE = []

import dpctl

from . import initialize
from .core import target
from .decorators import func, kernel

initialize.load_dpctl_sycl_interface()
