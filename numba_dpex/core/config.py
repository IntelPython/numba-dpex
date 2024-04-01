# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
The config options are meant to provide extra information and tweak optimization
configurations to help debug code generation issues.

There are two ways of setting these config options:

- Config options can be directly set programmatically, *e.g.*,

    .. code-block:: python

        from numba_dpex.core.config import DUMP_KERNEL_LLVM

        DUMP_KERNEL_LLVM = 1

- The options can also be set globally using environment flags. The name of the
  environment variable for every config option is annotated next to its
  definition.

    .. code-block:: bash

        export NUMBA_DPEX_DUMP_KERNEL_LLVM = 1

"""

from __future__ import annotations

import logging
import os
from typing import Annotated

from numba.core import config


def _readenv(name, ctor, default):
    """Read/write values from/into system environment variable list.

    This function is used to read and write values from (and into) system `env`.
    This is adapted from `process_environ()` function of `_EnvLoader` class in
    `numba/core/config.py`.

    Args:
        name (str): The name of the env variable.
        ctor (type): The type of the env variable.
        default (int,float,str): The default value of the env variable.

    Returns:
        int,float,string: The environment variable value of the specified type.
    """

    value = os.environ.get(name)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        logging.exception(
            "env variable %s defined but failed to parse '%s'" % (name, value)
        )
        return default


def __getattr__(name):
    """__getattr__ for numba_dpex's config module.

    This will be used to fallback to numba's config.

    Args:
        name (str): The name of the env variable.

    Returns:
        int,float,str: The environment variable value from numba.
    """
    return getattr(config, name)


SAVE_IR_FILES: Annotated[
    int,
    "Save the IR files (LLVM and SPIRV-V) generated for each kernel to"
    " current directory",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_SAVE_IR_FILES",
] = _readenv("NUMBA_DPEX_SAVE_IR_FILES", int, 0)

OFFLOAD_DIAGNOSTICS: Annotated[
    int,
    "Print diagnostic information for automatic offloading of parfor nodes "
    "to kernels",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_OFFLOAD_DIAGNOSTICS",
] = _readenv("NUMBA_DPEX_OFFLOAD_DIAGNOSTICS", int, 0)

DEBUG: Annotated[
    int,
    "Generates extra debug prints when set to a non-zero value",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_DEBUG",
] = _readenv("NUMBA_DPEX_DEBUG", int, config.DEBUG)

DEBUGINFO_DEFAULT: Annotated[
    int,
    "Compiles in the debug mode generating debug symbols in the compiler IR. "
    'It is a global way of setting the "debug" keyword for all '
    "numba_dpex.kernel and numba_dpex.device_func decorators "
    "used in a program.",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_DEBUGINFO",
] = _readenv("NUMBA_DPEX_DEBUGINFO", int, config.DEBUGINFO_DEFAULT)

DUMP_KERNEL_LLVM: Annotated[
    int,
    "Writes the optimized LLVM IR generated for a "
    "numba_dpex.kernel decorated function to current directory",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_DUMP_KERNEL_LLVM",
] = _readenv("NUMBA_DPEX_DUMP_KERNEL_LLVM", int, 0)

DUMP_KERNEL_LAUNCHER: Annotated[
    int,
    "Writes the optimized LLVM IR generated for every "
    "numba_dpex.call_kernel function to current directory",
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_DUMP_KERNEL_LAUNCHER",
] = _readenv("NUMBA_DPEX_DUMP_KERNEL_LAUNCHER", int, 0)

DEBUG_KERNEL_LAUNCHER: Annotated[
    int,
    "Enables debug printf messages inside the compiled module generated for a "
    "numba_dpex.call_kernel function."
    "default = 0",
    "ENVIRONMENT FLAG: NUMBA_DPEX_DEBUG_KERNEL_LAUNCHER",
] = _readenv("NUMBA_DPEX_DEBUG_KERNEL_LAUNCHER", int, 0)

BUILD_KERNEL_OPTIONS: Annotated[
    str,
    "Can use used to pass extra flags to the device driver compiler during "
    "kernel compilation. For available OpenCL options refer "
    "https://intel.github.io/llvm-docs/clang/ClangCommandLineReference.html#opencl-options",
    'default = ""',
    "ENVIRONMENT FLAG: NUMBA_DPEX_BUILD_KERNEL_OPTIONS",
] = _readenv("NUMBA_DPEX_BUILD_KERNEL_OPTIONS", str, "")

TESTING_SKIP_NO_DEBUGGING = _readenv(
    "NUMBA_DPEX_TESTING_SKIP_NO_DEBUGGING", int, 1
)

TESTING_LOG_DEBUGGING: Annotated[
    int,
    "Generates extra logs when using gdb to debug a kernel",
    "defaults = 0",
    "ENVIRONMENT_FLAG: NUMBA_DPEX_TESTING_LOG_DEBUGGING",
] = _readenv("NUMBA_DPEX_TESTING_LOG_DEBUGGING", int, DEBUG)

DPEX_OPT: Annotated[
    int,
    "Sets the optimization level globally for every function "
    "compiled by numba-dpex",
    "default = 2",
    "ENVIRONMENT_FLAG: NUMBA_DPEX_OPT",
] = _readenv("NUMBA_DPEX_OPT", int, 2)

INLINE_THRESHOLD: Annotated[
    int,
    "Sets the inlining-threshold level globally for every function "
    "compiled by numba-dpex. A higher value enables more aggressive inlining "
    "settings for the compiler. Note: Even if NUMBA_DPEX_INLINE_THRESHOLD is "
    'set to 0, many internal functions that are attributed "alwaysinline" '
    "will still get inlined.",
    "default = 2",
    "ENVIRONMENT_FLAG: NUMBA_DPEX_INLINE_THRESHOLD",
] = _readenv("NUMBA_DPEX_INLINE_THRESHOLD", int, 2)
