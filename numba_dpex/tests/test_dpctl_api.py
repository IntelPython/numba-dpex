# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import pytest

from numba_dpex.tests._helper import filter_strings


@pytest.mark.parametrize("filter_str", filter_strings)
def test_dpctl_api(filter_str):
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device):
        dpctl.lsplatform()
        dpctl.get_current_queue()
        dpctl.get_num_activated_queues()
        dpctl.is_in_device_context()
