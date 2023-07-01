# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.datamodel import models

from numba_dpex.core.datamodel.models import (
    DpnpNdArrayModel,
    dpex_data_model_manager,
)
from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray


def test_model_for_DpnpNdArray():
    """Test that model is registered for DpnpNdArray instances.

    The model for DpnpNdArray is dpex's ArrayModel.

    """

    model = dpex_data_model_manager.lookup(
        DpnpNdArray(ndim=1, dtype=types.float64, layout="C")
    )
    assert isinstance(model, DpnpNdArrayModel)


def test_dpnp_ndarray_Model():
    """Test for dpnp_ndarray_Model.

    It is a subclass of models.StructModel and models.ArrayModel.
    """

    assert issubclass(DpnpNdArrayModel, models.StructModel)
