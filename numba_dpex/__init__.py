# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The numba-dpex extension module adds data-parallel offload support to Numba.
"""
import numba.testing

from numba_dpex.interop import asarray
from numba_dpex.retarget import offload_to_sycl_device

from . import config

if config.HAS_NON_HOST_DEVICE:
    from .device_init import *
else:
    raise ImportError("No non-host SYCL device found to execute kernels.")


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = ["offload_to_sycl_device"]
