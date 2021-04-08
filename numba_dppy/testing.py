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
