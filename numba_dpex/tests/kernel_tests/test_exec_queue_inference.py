# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import pytest
from numba.core import config

import numba_dpex as dpex
from numba_dpex import Range
from numba_dpex.core.exceptions import ExecutionQueueInferenceError


@dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def add(a, b, c):
    c[0] = b[0] + a[0]


def test_successful_execution_queue_inference():
    """
    Tests if KernelDispatcher successfully infers the execution queue for the
    kernel.
    """

    q = dpctl.SyclQueue()
    a = dpnp.ones(100, sycl_queue=q)
    b = dpnp.ones_like(a, sycl_queue=q)
    c = dpnp.zeros_like(a, sycl_queue=q)
    r = Range(100)

    current_captured_error_style = config.CAPTURED_ERRORS
    config.CAPTURED_ERRORS = "new_style"

    try:
        dpex.call_kernel(add, r, a, b, c)
    except:
        pytest.fail("Unexpected error when calling kernel")

    config.CAPTURED_ERRORS = current_captured_error_style

    assert c[0] == b[0] + a[0]


def test_execution_queue_inference_error():
    """
    Tests if KernelDispatcher successfully raised ExecutionQueueInferenceError
    when dpnp.ndarray arguments do not share the same dpctl.SyclQueue
    instance.
    """

    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()
    a = dpnp.ones(100, sycl_queue=q1)
    b = dpnp.ones_like(a, sycl_queue=q2)
    c = dpnp.zeros_like(a, sycl_queue=q1)
    r = Range(100)

    current_captured_error_style = config.CAPTURED_ERRORS
    config.CAPTURED_ERRORS = "new_style"

    with pytest.raises(ExecutionQueueInferenceError):
        dpex.call_kernel(add, r, a, b, c)

    config.CAPTURED_ERRORS = current_captured_error_style


def test_error_when_no_array_args():
    """
    Tests if KernelDispatcher successfully raised ExecutionQueueInferenceError
    when no dpnp.ndarray arguments were passed to a kernel.
    """
    a = 1
    b = 2
    c = 3
    r = Range(100)

    from numba.core import config

    current_captured_error_style = config.CAPTURED_ERRORS
    config.CAPTURED_ERRORS = "new_style"

    with pytest.raises(ExecutionQueueInferenceError):
        dpex.call_kernel(add, r, a, b, c)

    config.CAPTURED_ERRORS = current_captured_error_style
