# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import dpctl
from numba.core import config


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
        logging.exception(
            "environ %s defined but failed to parse '%s'" % (name, value)
        )
        return default


def __getattr__(name):
    """Fallback to Numba config"""
    return getattr(config, name)


# Ensure dpctl can create a default sycl device.
# Set this config flag based on if dpctl is found or not.
# The config flags is used elsewhere inside Numba.
try:
    HAS_NON_HOST_DEVICE = dpctl.select_default_device()
except Exception:
    logging.exception(
        "dpctl could not find any non-host SYCL device on the system. "
        + "A non-host SYCL device is required to use numba_dpex."
    )

# To save intermediate files generated by th compiler
SAVE_IR_FILES = _readenv("NUMBA_DPEX_SAVE_IR_FILES", int, 0)

# Turn SPIRV-VALIDATION ON/OFF switch
SPIRV_VAL = _readenv("NUMBA_DPEX_SPIRV_VAL", int, 0)

# Dump offload diagnostics
OFFLOAD_DIAGNOSTICS = _readenv("NUMBA_DPEX_OFFLOAD_DIAGNOSTICS", int, 0)

# Emit debug info
DEBUG = _readenv("NUMBA_DPEX_DEBUG", int, config.DEBUG)
DEBUGINFO_DEFAULT = _readenv(
    "NUMBA_DPEX_DEBUGINFO", int, config.DEBUGINFO_DEFAULT
)

# Emit LLVM assembly language format(.ll)
DUMP_KERNEL_LLVM = _readenv(
    "NUMBA_DPEX_DUMP_KERNEL_LLVM", int, config.DUMP_OPTIMIZED
)

# configs for caching
# To see the debug messages for the caching.
# Execute like:
#   NUMBA_DPEX_DEBUG_CACHE=1 python <code>
DEBUG_CACHE = _readenv("NUMBA_DPEX_DEBUG_CACHE", int, 0)
# This is a global flag to turn the caching on/off,
# regardless of whatever has been specified in Dispatcher.
# Useful for debugging. Execute like:
#   NUMBA_DPEX_ENABLE_CACHE=0 python <code>
# to turn off the caching globally.
ENABLE_CACHE = _readenv("NUMBA_DPEX_ENABLE_CACHE", int, 1)
# Capacity of the cache, execute it like:
#   NUMBA_DPEX_CACHE_SIZE=20 python <code>
CACHE_SIZE = _readenv("NUMBA_DPEX_CACHE_SIZE", int, 128)

TESTING_SKIP_NO_DPNP = _readenv("NUMBA_DPEX_TESTING_SKIP_NO_DPNP", int, 0)
TESTING_SKIP_NO_DEBUGGING = _readenv(
    "NUMBA_DPEX_TESTING_SKIP_NO_DEBUGGING", int, 1
)
TESTING_LOG_DEBUGGING = _readenv("NUMBA_DPEX_TESTING_LOG_DEBUGGING", int, DEBUG)
# Flag to turn on the ConstantSizeStaticLocalMemoryPass in the kernel pipeline.
# The pass is turned off by default.
STATIC_LOCAL_MEM_PASS = _readenv("NUMBA_DPEX_STATIC_LOCAL_MEM_PASS", int, 0)
