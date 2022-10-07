# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# Re export
from .ocl.stubs import (
    CLK_GLOBAL_MEM_FENCE,
    CLK_LOCAL_MEM_FENCE,
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

from . import initialize, target
from .decorators import autojit, func, kernel


def is_available():
    """Returns a boolean indicating if dpctl could find a default device.

    A ValueError is thrown by dpctl if no default device is found and it
    implies that numba_dpex cannot create a SYCL queue to compile kernels.

    Returns:
        bool: True if a default SYCL device is found, otherwise False.
    """
    try:
        d = dpctl.select_default_device()
        return not d.is_host
    except ValueError:
        return False


initialize.load_dpctl_sycl_interface()
