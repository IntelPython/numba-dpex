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

import os
import warnings
from packaging import version


class DpctlMinimumVersionRequiredError(Exception):
    """
    A ``DpctlMinimumVersionRequiredError`` indicates that the version of dpctl
    does not satisfy the minimum version requirement.

    """

    pass


# Check for dpctl 0.8.0 or higher in the system.
_dpctl_found = False
try:
    import dpctl

    # Versions of dpctl lower than 0.8.0 are not compatible with current main
    # of numba_dppy.
    if version.parse(dpctl.__version__) < version.parse("0.8.*"):
        raise DpctlMinimumVersionRequiredError

    # For the Numba_dppy extension to work, we should have at least one
    # non-host SYCL device installed.
    # FIXME: In future, we should support just the host device.
    if not dpctl.select_default_device().is_host:
        _dpctl_found = True
    else:
        msg = "dpctl could not find any non-host SYCL device on the system. "
        msg += "A non-host SYCL device is required to use numba_dppy."
        warnings.warn(msg, UserWarning)
except DpctlMinimumVersionRequiredError:
    msg = "numba_dppy is not compatible with " + dpctl.__version__ + "."
    msg += " Install dpctl 0.8.* or higher."
    warnings.warn(msg, UserWarning)
except:
    msg = "Please install dpctl 0.8.* or higher."
    warnings.warn(msg, UserWarning)

# Set this config flag based on if dpctl is found or not. The config flags is
# used elsewhere inside Numba.
dppy_present = _dpctl_found


def _readenv(name, ctor, default):
    """Original version from numba\core\config.py
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


# Save intermediate files being generated by DPPY
SAVE_IR_FILES = _readenv("NUMBA_DPPY_SAVE_IR_FILES", int, 0)

# Turn SPIRV-VALIDATION ON/OFF switch
SPIRV_VAL = _readenv("NUMBA_DPPY_SPIRV_VAL", int, 0)

# Dump offload diagnostics
OFFLOAD_DIAGNOSTICS = _readenv("NUMBA_DPPY_OFFLOAD_DIAGNOSTICS", int, 0)

FALLBACK_ON_CPU = _readenv("NUMBA_DPPY_FALLBACK_ON_CPU", int, 1)

# Activate Native floating point atomcis support for supported devices.
# Requires llvm-spirv supporting the FP atomics extensio
NATIVE_FP_ATOMICS = _readenv("NUMBA_DPPY_ACTIVATE_ATOMCIS_FP_NATIVE", int, 0)
LLVM_SPIRV_ROOT = _readenv("NUMBA_DPPY_LLVM_SPIRV_ROOT", str, "")
# Emit debug info
DEBUG = _readenv("NUMBA_DPPY_DEBUG", int, 0)
DEBUGINFO = _readenv("NUMBA_DPPY_DEBUGINFO", int, 0)
