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

import sys
import contextlib

import unittest
import dpctl
from numba_dppy.context_manager import offload_to_sycl_device
from numba.tests.support import (
    captured_stdout,
    redirect_c_stdout,
)

import numba_dppy
from numba_dppy import config


def has_gpu_queues(backend="opencl"):
    """
    Checks if dpctl is able to select a GPU device that defaults to
    an OpenCL GPU.
    """
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))


def has_cpu_queues(backend="opencl"):
    """
    Checks if dpctl is able to select a CPU device that defaults to
    an OpenCL CPU.
    """
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))


def has_sycl_platforms():
    """
    Checks if dpctl is able to identify a non-host SYCL platform.
    """
    platforms = dpctl.get_platforms()
    for p in platforms:
        if not p.backend is dpctl.backend_type.host:
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


def skip_test(device_type):
    skip = False
    try:
        with dpctl.device_context(device_type):
            pass
    except Exception:
        skip = True

    if not skip:
        if platform_not_supported(device_type):
            skip = True

    return skip


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


@contextlib.contextmanager
def captured_dppy_stdout():
    """
    Return a minimal stream-like object capturing the text output of dppy
    """
    # Prevent accidentally capturing previously output text
    sys.stdout.flush()

    import numba_dppy, numba_dppy as dppy

    with redirect_c_stdout() as stream:
        yield DPPYTextCapture(stream)


def _id(obj):
    return obj


def expectedFailureIf(condition):
    """
    Expected failure for a test if the condition is true.
    """
    if condition:
        return unittest.expectedFailure
    return _id


def ensure_dpnp():
    try:
        from numba_dppy.dpnp_glue import dpnp_fptr_interface as dpnp_glue

        return True
    except:
        return False


@contextlib.contextmanager
def dpnp_debug():
    import numba_dppy.dpnp_glue as dpnp_lowering

    old, dpnp_lowering.DEBUG = dpnp_lowering.DEBUG, 1
    yield
    dpnp_lowering.DEBUG = old


@contextlib.contextmanager
def assert_dpnp_implementaion():
    from numba.tests.support import captured_stdout

    with captured_stdout() as stdout, dpnp_debug():
        yield

    assert "dpnp implementation" in stdout.getvalue(), "dpnp implementation is not used"


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
