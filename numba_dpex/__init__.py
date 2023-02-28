# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The numba-dpex extension module adds data-parallel offload support to Numba.
"""

from numba.core import ir_utils
from numba.np import arrayobj
from numba.np.ufunc import array_exprs

from .numba_patches.patch_mk_alloc import _mk_alloc  # isort: skip

ir_utils.mk_alloc = _mk_alloc  # isort: skip
import numba_dpex.core.dpjit_dispatcher
import numba_dpex.core.offload_dispatcher

# Initialize the _dpexrt_python extension
import numba_dpex.core.runtime
import numba_dpex.core.targets.dpjit_target

# Re-export types itself
import numba_dpex.core.types as types
from numba_dpex.core.kernel_interface.indexers import NdRange, Range

# Re-export all type names
from numba_dpex.core.types import *
from numba_dpex.retarget import offload_to_sycl_device

from . import config
from ._version import get_versions
from .numba_patches.patch_empty_nd_impl import _dpex_empty_nd_impl
from .numba_patches.patch_is_ufunc import _dpex_is_ufunc

arrayobj._empty_nd_impl = _dpex_empty_nd_impl
array_exprs._is_ufunc = _dpex_is_ufunc


if config.HAS_NON_HOST_DEVICE:
    from .device_init import *
else:
    raise ImportError("No non-host SYCL device found to execute kernels.")


__version__ = get_versions()["version"]
del get_versions

__all__ = ["offload_to_sycl_device"] + types.__all__ + ["Range", "NdRange"]
