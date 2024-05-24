#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
import shutil
from functools import cache

import dpctl
import dpnp
import pytest

from numba_dpex import config, dpjit


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


def has_cpu():
    """
    Checks if dpctl is able to select any CPU device.
    """
    return bool(dpctl.get_num_devices(device_type="cpu"))


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


def is_gen12():
    """Checks if the default device is an Intel Gen12 (Xe) GPU."""
    device_name = dpctl.SyclDevice().name
    if "Gen12" in device_name:
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

skip_no_gdb = pytest.mark.skipif(
    config.TESTING_SKIP_NO_DEBUGGING or not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
)

decorators = [
    pytest.param(dpjit, id="dpjit"),
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


def get_complex_dtypes(device=None):
    """
    Build a list of complex types supported by DPNP based on device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add complex types
    dtypes = [dpnp.complex64]
    if dev.has_aspect_fp64:
        dtypes.append(dpnp.complex128)
    return dtypes


def get_int_dtypes(device=None):
    """
    Build a list of integer types supported by DPNP based on device capabilities.
    """

    return [dpnp.int32, dpnp.int64]


def get_float_dtypes(no_float16=True, device=None):
    """
    Build a list of floating types supported by DPNP based on device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add floating types
    dtypes = []
    if not no_float16 and dev.has_aspect_fp16:
        dtypes.append(dpnp.float16)

    dtypes.append(dpnp.float32)
    if dev.has_aspect_fp64:
        dtypes.append(dpnp.float64)
    return dtypes


def get_float_complex_dtypes(no_float16=True, device=None):
    """
    Build a list of floating and complex types supported by DPNP based on device capabilities.
    """

    dtypes = get_float_dtypes(no_float16, device)
    dtypes.extend(get_complex_dtypes(device))
    return dtypes


def get_all_dtypes(
    no_bool=False,
    no_int=False,
    no_float16=True,
    no_float=False,
    no_complex=False,
    no_none=False,
    device=None,
):
    """
    Build a list of types supported by DPNP based on input flags and device capabilities.
    """

    dev = dpctl.select_default_device() if device is None else device

    # add boolean type
    dtypes = [dpnp.bool] if not no_bool else []

    # add integer types
    if not no_int:
        dtypes.extend(get_int_dtypes(device=dev))

    # add floating types
    if not no_float:
        dtypes.extend(get_float_dtypes(no_float16=no_float16, device=dev))

    # add complex types
    if not no_complex:
        dtypes.extend(get_complex_dtypes(device=dev))

    # add None value to validate a default dtype
    if not no_none:
        dtypes.append(None)
    return dtypes


def get_queue_or_skip(args=tuple()):
    try:
        q = dpctl.SyclQueue(*args)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"Queue could not be created from {args}")
    return q


def skip_if_dtype_not_supported(dt, q_or_dev):
    import dpctl.tensor as dpt

    dt = dpt.dtype(dt)
    if type(q_or_dev) is dpctl.SyclQueue:
        dev = q_or_dev.sycl_device
    elif type(q_or_dev) is dpctl.SyclDevice:
        dev = q_or_dev
    else:
        raise TypeError(
            "Expected dpctl.SyclQueue or dpctl.SyclDevice, "
            f"got {type(q_or_dev)}"
        )
    dev_has_dp = dev.has_aspect_fp64
    if dev_has_dp is False and dt in [dpt.float64, dpt.complex128]:
        pytest.skip(
            f"{dev.name} does not support double precision floating point types"
        )
    dev_has_hp = dev.has_aspect_fp16
    if dev_has_hp is False and dt in [
        dpt.float16,
    ]:
        pytest.skip(
            f"{dev.name} does not support half precision floating point type"
        )


def num_required_arguments(func):
    """Returns number of required arguments of the functions. Does not work
    with kwargs arguments."""
    if func == dpnp.true_divide:
        func = dpnp.divide

    sig = inspect.signature(func)
    params = sig.parameters
    required_args = [
        p
        for p in params
        if params[p].default == inspect._empty and p != "kwargs"
    ]

    return len(required_args)
