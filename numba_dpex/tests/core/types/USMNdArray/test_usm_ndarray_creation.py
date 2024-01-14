# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import pytest

from numba_dpex.core.types import DpctlSyclQueue, USMNdArray

"""Negative tests for expected exceptions raised during USMNdArray creation.

"""


def test_default_type_construction():
    """Tests call USMNdArray constructor with no device or queue args."""
    usma = USMNdArray(1, queue=None)

    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"

    default_device = dpctl.SyclDevice()
    cached_queue = dpctl._sycl_queue_manager.get_device_cached_queue(
        default_device
    )

    assert usma.device == default_device.filter_string
    if usma.queue != cached_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as cached against the device."
        )


def test_type_creation_with_device():
    """Tests creating a USMNdArray with a device arg and no queue"""

    default_device_str = dpctl.SyclDevice().filter_string

    usma = USMNdArray(1, device=default_device_str, queue=None)

    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"

    assert usma.device == default_device_str

    cached_queue = dpctl._sycl_queue_manager.get_device_cached_queue(
        default_device_str
    )

    if usma.queue != cached_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as cached "
            "against the device."
        )


def test_type_creation_with_queue():
    """Tests creating a USMNdArray with a queue arg and no device"""
    ty_queue = DpctlSyclQueue(dpctl.SyclQueue())
    usma = USMNdArray(1, queue=ty_queue)

    assert usma.ndim == 1
    assert usma.layout == "C"
    assert usma.addrspace == 1
    assert usma.usm_type == "device"

    assert usma.device == ty_queue.sycl_device
    if usma.queue != ty_queue:
        pytest.xfail(
            "Returned queue does not have the same queue as the one passed "
            "to the dpnp function."
        )


def test_exception_when_both_device_and_queue_arg_specified():
    """Tests if TypeError is raised when both queue and device specified"""

    queue = dpctl.SyclQueue()
    with pytest.raises(TypeError):
        USMNdArray(1, device="cpu", queue=queue)


def test_improper_queue_type():
    """Tests if TypeError is raised if queue argument is of invalid type"""

    with pytest.raises(TypeError):
        USMNdArray(1, queue="cpu")


def test_improper_device_type():
    """Tests if TypeError is raised if device argument is of invalid type"""

    with pytest.raises(TypeError):
        USMNdArray(1, device=0)
