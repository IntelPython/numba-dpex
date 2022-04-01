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

import imp
import os
import warnings

from numba.core import config


def _ensure_dpctl():
    """
    Make sure dpctl has supported versions.
    """
    from numba_dppy.dpctl_support import dpctl_version

    if dpctl_version < (0, 8):
        raise ImportError("numba_dpex needs dpctl 0.8 or greater")


def _dpctl_has_non_host_device():
    """
    Make sure dpctl has non-host SYCL devices on the system.
    """
    import dpctl

    # For the numba_dpex extension to work, we should have at least one
    # non-host SYCL device installed.
    # FIXME: In future, we should support just the host device.
    if not dpctl.select_default_device().is_host:
        return True
    else:
        msg = "dpctl could not find any non-host SYCL device on the system. "
        msg += "A non-host SYCL device is required to use numba_dpex."
        warnings.warn(msg, UserWarning)
        return False


_ensure_dpctl()

# Set this config flag based on if dpctl is found or not. The config flags is
# used elsewhere inside Numba.
HAS_NON_HOST_DEVICE = _dpctl_has_non_host_device()


def _readenv(name, ctor, default):
    """Original version from numba/core/config.py
    class _EnvReloader():
        ...
        def process_environ():
            def _readenv(): ...
    """
    value = os.environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        import warnings

        warnings.warn(
            "environ %s defined but failed to parse '%s'" % (name, value),
            RuntimeWarning,
        )
        return default


def __getattr__(name):
    """Fallback to Numba config"""
    return getattr(config, name)


# To save intermediate files generated by th compiler
SAVE_IR_FILES = _readenv("NUMBA_DPPY_SAVE_IR_FILES", int, 0)

# Turn SPIRV-VALIDATION ON/OFF switch
SPIRV_VAL = _readenv("NUMBA_DPPY_SPIRV_VAL", int, 0)

# Dump offload diagnostics
OFFLOAD_DIAGNOSTICS = _readenv("NUMBA_DPPY_OFFLOAD_DIAGNOSTICS", int, 0)

FALLBACK_ON_CPU = _readenv("NUMBA_DPPY_FALLBACK_ON_CPU", int, 1)

# Activate Native floating point atomcis support for supported devices.
# Requires llvm-spirv supporting the FP atomics extension
NATIVE_FP_ATOMICS = _readenv("NUMBA_DPPY_ACTIVATE_ATOMICS_FP_NATIVE", int, 0)
LLVM_SPIRV_ROOT = _readenv("NUMBA_DPPY_LLVM_SPIRV_ROOT", str, "")
# Emit debug info
DEBUG = _readenv("NUMBA_DPPY_DEBUG", int, config.DEBUG)
DEBUGINFO_DEFAULT = _readenv(
    "NUMBA_DPPY_DEBUGINFO", int, config.DEBUGINFO_DEFAULT
)

TESTING_SKIP_NO_DPNP = _readenv("NUMBA_DPPY_TESTING_SKIP_NO_DPNP", int, 0)
TESTING_SKIP_NO_OPENCL_CPU = _readenv(
    "NUMBA_DPEX_TESTING_SKIP_NO_OPENCL_CPU", int, 0
)
TESTING_SKIP_NO_OPENCL_GPU = _readenv(
    "NUMBA_DPEX_TESTING_SKIP_NO_OPENCL_GPU", int, 0
)
TESTING_SKIP_NO_LEVEL_ZERO_GPU = _readenv(
    "NUMBA_DPEX_TESTING_SKIP_NO_LEVEL_ZERO_GPU", int, 0
)
TESTING_SKIP_NO_DEBUGGING = _readenv(
    "NUMBA_DPPY_TESTING_SKIP_NO_DEBUGGING", int, 1
)
TESTING_LOG_DEBUGGING = _readenv("NUMBA_DPPY_TESTING_LOG_DEBUGGING", int, DEBUG)
