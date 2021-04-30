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
import sys

from numba.core import config
import unittest
from numba.tests.support import (
    captured_stdout,
    redirect_c_stdout,
)
import numba_dppy


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
    old_debug = numba_dppy.compiler.DEBUG
    numba_dppy.compiler.DEBUG = 1

    with captured_stdout() as stdout:
        yield

    numba_dppy.compiler.DEBUG = old_debug

    got_parfor_offloaded = stdout.getvalue().count(
        "Parfor lowered to specified SYCL device"
    )
    assert parfor_offloaded == got_parfor_offloaded, (
        "Expected %d parfor(s) to be auto offloaded, instead got %d parfor(s) auto offloaded"
        % (parfor_offloaded, got_parfor_offloaded)
    )

    got_parfor_offloaded_failure = stdout.getvalue().count(
        "Failed to lower parfor on SYCL device. Falling back to default CPU parallelization."
    )
    assert parfor_offloaded_failure == got_parfor_offloaded_failure, (
        "Expected %d parfor(s) to be not auto offloaded, instead got %d parfor(s) not auto offloaded"
        % (parfor_offloaded_failure, got_parfor_offloaded_failure)
    )
