#! /usr/bin/env python
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

import contextlib
import shutil

import dpctl
import pytest
from numba.tests.support import captured_stdout

from numba_dppy import config
from numba_dppy.numba_support import numba_version


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


skip_no_opencl_gpu = pytest.mark.skipif(
    not has_opencl_gpu() and config.TESTING_SKIP_NO_OPENCL_GPU,
    reason="No opencl GPU platforms available",
)
skip_no_opencl_cpu = pytest.mark.skipif(
    not has_opencl_cpu() and config.TESTING_SKIP_NO_OPENCL_CPU,
    reason="No opencl CPU platforms available",
)
skip_no_level_zero_gpu = pytest.mark.skipif(
    not has_level_zero() and config.TESTING_SKIP_NO_LEVEL_ZERO_GPU,
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

skip_no_numba055 = pytest.mark.skipif(
    numba_version < (0, 55), reason="Need Numba 0.55 or higher"
)

skip_no_gdb = pytest.mark.skipif(
    config.TESTING_SKIP_NO_DEBUGGING and not shutil.which("gdb-oneapi"),
    reason="Intel® Distribution for GDB* is not available",
)


@contextlib.contextmanager
def override_config(name, value, config=config):
    """
    Extends `numba/tests/support.py:override_config()` with argument `config`
    which is `numba_dppy.config` by default.
    """
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)


def _id(obj):
    return obj


def _ensure_dpnp():
    try:
        from numba_dppy.dpnp_iface import dpnp_fptr_interface as dpnp_iface

        return True
    except ImportError:
        if config.TESTING_SKIP_NO_DPNP:
            return False
        else:
            pytest.fail("DPNP is not available")


skip_no_dpnp = pytest.mark.skipif(
    not _ensure_dpnp(), reason="DPNP is not available"
)


@contextlib.contextmanager
def dpnp_debug():
    import numba_dppy.dpnp_iface as dpnp_lowering

    old, dpnp_lowering.DEBUG = dpnp_lowering.DEBUG, 1
    yield
    dpnp_lowering.DEBUG = old


@contextlib.contextmanager
def assert_dpnp_implementaion():
    from numba.tests.support import captured_stdout

    with captured_stdout() as stdout, dpnp_debug():
        yield

    assert (
        "dpnp implementation" in stdout.getvalue()
    ), "dpnp implementation is not used"


@contextlib.contextmanager
def assert_auto_offloading(parfor_offloaded=1, parfor_offloaded_failure=0):
    """
    If ``parfor_offloaded`` is not provided this context_manager
    will check for 1 occurrance of success message. Developers
    can always specify how many parfor offload success message
    is expected.
    If ``parfor_offloaded_failure`` is not provided the default
    behavior is to expect 0 failure message, in other words, we
    expect all parfors present in the code to be successfully
    offloaded to GPU.
    """
    old_debug = config.DEBUG
    config.DEBUG = 1

    with captured_stdout() as stdout:
        yield

    config.DEBUG = old_debug

    got_parfor_offloaded = stdout.getvalue().count("Parfor offloaded to")
    assert parfor_offloaded == got_parfor_offloaded, (
        "Expected %d parfor(s) to be auto offloaded, instead got %d parfor(s) auto offloaded"
        % (parfor_offloaded, got_parfor_offloaded)
    )

    got_parfor_offloaded_failure = stdout.getvalue().count(
        "Failed to offload parfor to"
    )
    assert parfor_offloaded_failure == got_parfor_offloaded_failure, (
        "Expected %d parfor(s) to be not auto offloaded, instead got %d parfor(s) not auto offloaded"
        % (parfor_offloaded_failure, got_parfor_offloaded_failure)
    )
