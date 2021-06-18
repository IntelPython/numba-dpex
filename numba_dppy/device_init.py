# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Re export
from .ocl.stubs import (
    get_global_id,
    get_global_size,
    get_local_id,
    get_local_size,
    get_group_id,
    get_work_dim,
    get_num_groups,
    barrier,
    mem_fence,
    sub_group_barrier,
    atomic,
    local,
    CLK_LOCAL_MEM_FENCE,
    CLK_GLOBAL_MEM_FENCE,
)

"""
We are importing dpnp stub module to make Numba recognize the
module when we rename Numpy functions.
"""
from .dpnp_glue.stubs import dpnp

DEFAULT_LOCAL_SIZE = []

from . import initialize

from .decorators import kernel, func, autojit
import dpctl
from . import target


def is_available():
    """Returns a boolean indicating if dpctl could find a default device.

    A valueError is thrown by dpctl if no default device is found and it
    implies that numba-dppy cannot create a SYCL queue to compile kernels.

    Returns:
        bool: True if a default SYCL device is found, otherwise False.
    """
    try:
        d = dpctl.select_default_device()
        return not d.is_host
    except ValueError:
        return False


initialize.load_dpctl_sycl_interface()
