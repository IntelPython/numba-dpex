# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.datamodel import default_manager, models

from numba_dpex.core.datamodel.models import (
    DpnpNdArrayModel,
    USMArrayModel,
    dpex_data_model_manager,
)
from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray


def test_model_for_DpnpNdArray():
    """Test the datamodel for DpnpNdArray that is registered with numba's
    default datamodel manager and numba_dpex's kernel data model manager.
    """
    dpnp_ndarray = DpnpNdArray(ndim=1, dtype=types.float64, layout="C")
    model = dpex_data_model_manager.lookup(dpnp_ndarray)
    assert isinstance(model, USMArrayModel)
    default_model = default_manager.lookup(dpnp_ndarray)
    assert isinstance(default_model, DpnpNdArrayModel)


def test_dpnp_ndarray_Model():
    """Test for dpnp_ndarray_Model.

    It is a subclass of models.StructModel and models.ArrayModel.
    """

    assert issubclass(DpnpNdArrayModel, models.StructModel)
