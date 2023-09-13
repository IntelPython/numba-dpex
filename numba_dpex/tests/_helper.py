#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import shutil
from functools import cache

import dpctl
import pytest

from numba_dpex import config, dpjit, numba_sem_version


@cache
def has_numba_mlir():
    try:
        import numba_mlir
    except ImportError:
        return False

    return True


def has_opencl_gpu():
    """
    Checks if dpctl is able to select an OpenCL GPU device.
    """
    return bool(dpctl.get_num_devices(backend="opencl", device_type="gpu"))


def has_opencl_cpu():
    """
    Checks if dpctl is able to select an OpenCL CPU device.
    """
    return bool(dpctl.get_num_devices(backend="opencl", device_type="cpu"))


def has_level_zero():
    """
    Checks if dpctl is able to select a Level Zero GPU device.
    """
    return bool(dpctl.get_num_devices(backend="level_zero", device_type="gpu"))


def has_sycl_platforms():
    """
    Checks if dpctl is able to identify a non-host SYCL platform.
    """
    platforms = dpctl.get_platforms()
    for p in platforms:
        if p.backend is not dpctl.backend_type.host:
            return True
    return False


def is_gen12(device_type):
    with dpctl.device_context(device_type):
        q = dpctl.get_current_queue()
        device = q.get_sycl_device()
        name = device.name
        if "Gen12" in name:
            return True

        return False


def platform_not_supported(device_type):
    import platform

    platform = platform.system()
    device = device_type.split(":")[0]

    if device == "level_zero" and platform == "Windows":
        return True

    return False


def is_windows():
    import platform

    return platform.system() == "Windows"


skip_windows = pytest.mark.skipif(is_windows(), reason="Skip on Windows")

skip_no_opencl_gpu = pytest.mark.skipif(
    not has_opencl_gpu(),
    reason="No opencl GPU platforms available",
)
skip_no_opencl_cpu = pytest.mark.skipif(
    not has_opencl_cpu(),
    reason="No opencl CPU platforms available",
)
skip_no_level_zero_gpu = pytest.mark.skipif(
    not has_level_zero(),
    reason="No level-zero GPU platforms available",
)
skip_no_numba_mlir = pytest.mark.skipif(
    not has_numba_mlir(),
    reason="numba-mlir package is not availabe",
)

filter_strings = [
    pytest.param("level_zero:gpu:0", marks=skip_no_level_zero_gpu),
    pytest.param("opencl:gpu:0", marks=skip_no_opencl_gpu),
    pytest.param("opencl:cpu:0", marks=skip_no_opencl_cpu),
]

mark_freeze = pytest.mark.skip(reason="Freeze")
mark_seg_fault = pytest.mark.skip(reason="Segmentation fault")

filter_strings_with_skips_for_opencl = [
    pytest.param("level_zero:gpu:0", marks=skip_no_level_zero_gpu),
    pytest.param("opencl:gpu:0", marks=mark_freeze),
    pytest.param("opencl:cpu:0", marks=mark_seg_fault),
]

filter_strings_opencl_gpu = [
    pytest.param("opencl:gpu:0", marks=skip_no_opencl_gpu),
]

filter_strings_level_zero_gpu = [
    pytest.param("level_zero:gpu:0", marks=skip_no_level_zero_gpu),
]

skip_no_numba056 = pytest.mark.skipif(
    numba_sem_version < (0, 56), reason="Need Numba 0.56 or higher"
)

skip_no_gdb = pytest.mark.skipif(
    config.TESTING_SKIP_NO_DEBUGGING and not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
)


decorators = [
    pytest.param(dpjit, id="dpjit"),
    pytest.param(
        dpjit(use_mlir=True), id="dpjit_mlir", marks=skip_no_numba_mlir
    ),
]


@contextlib.contextmanager
def override_config(name, value, config=config):
    """
    Extends `numba/tests/support.py:override_config()` with argument `config`
    which is `numba_dpex.config` by default.
    """
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)


def _id(obj):
    return obj
