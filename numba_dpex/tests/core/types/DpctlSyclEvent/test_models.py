# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
from numba import types
from numba.core.datamodel import default_manager, models

from numba_dpex.core.datamodel.models import (
    SyclEventModel,
    dpex_data_model_manager,
)
from numba_dpex.core.types.dpctl_types import DpctlSyclEvent


def test_model_for_DpctlSyclEvent():
    """Test the datamodel for DpctlSyclEvent that is registered with numba's
    default datamodel manager and numba_dpex's kernel data model manager.
    """
    sycl_event = DpctlSyclEvent(dpctl.SyclEvent())
    model = dpex_data_model_manager.lookup(sycl_event)
    assert isinstance(model, SyclEventModel)
    default_model = default_manager.lookup(sycl_event)
    assert isinstance(default_model, SyclEventModel)


def test_sycl_event_Model():
    """Test for sycl_event_Model.

    It is a subclass of models.StructModel and models.ArrayModel.
    """

    assert issubclass(SyclEventModel, models.StructModel)