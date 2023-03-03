# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The numba-dpex extension module adds data-parallel offload support to Numba.
"""

from numba.np.ufunc import array_exprs

import numba_dpex.core.dpjit_dispatcher
import numba_dpex.core.offload_dispatcher

from .numba_patches.patch_is_ufunc import _dpex_is_ufunc

array_exprs._is_ufunc = _dpex_is_ufunc

# Initialize the _dpexrt_python extension
import numba_dpex.core.runtime  # noqa: E402
import numba_dpex.core.targets.dpjit_target  # noqa: E402

# Re-export types itself
import numba_dpex.core.types as types  # noqa: E402
from numba_dpex.core.kernel_interface.indexers import (  # noqa: E402
    NdRange,
    Range,
)

# Re-export all type names
from numba_dpex.core.types import *  # noqa: E402
from numba_dpex.retarget import offload_to_sycl_device  # noqa: E402

from . import config  # noqa: E402

if config.HAS_NON_HOST_DEVICE:
    from .device_init import *  # noqa: E402
else:
    raise ImportError("No non-host SYCL device found to execute kernels.")


from ._version import get_versions  # noqa: E402

__version__ = get_versions()["version"]
del get_versions

__all__ = ["offload_to_sycl_device"] + types.__all__ + ["Range", "NdRange"]
