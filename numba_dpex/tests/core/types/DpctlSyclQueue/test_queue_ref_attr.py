# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl

from numba_dpex.tests.core.types.DpctlSyclQueue._helper import are_queues_equal


def test_queue_ref_access_in_dpjit():
    """Tests if we can access the queue_ref attribute of a dpctl.SyclQueue
    PyObject inside dpjit and pass it to a native C function, in this case
    dpctl's libsyclinterface's DPCTLQueue_AreEq.

    Checks if the result of queue equality check done inside dpjit is the
    same as when done in Python.
    """

    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    expected = q1 == q2
    actual = are_queues_equal(q1, q2)

    assert expected == actual

    d = dpctl.SyclDevice()
    cq1 = dpctl._sycl_queue_manager.get_device_cached_queue(d)
    cq2 = dpctl._sycl_queue_manager.get_device_cached_queue(d)

    expected = cq1 == cq2
    actual = are_queues_equal(cq1, cq2)

    assert expected == actual
