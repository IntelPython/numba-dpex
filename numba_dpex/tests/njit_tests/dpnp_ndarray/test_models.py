# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.datamodel import default_manager, models
from numba.core.datamodel.models import ArrayModel

from numba_dpex.core.types.dpnp_ndarray_type import DpnpNdArray


def test_model_for_DpnpNdArray():
    """Test that model is registered for DpnpNdArray instances.

    The model for DpnpNdArray is dpex's ArrayModel.

    """

    model = default_manager.lookup(
        DpnpNdArray(ndim=1, dtype=types.float64, layout="C")
    )
    assert isinstance(model, ArrayModel)


def test_dpnp_ndarray_Model():
    """Test for dpnp_ndarray_Model.

    It is a subclass of models.StructModel and models.ArrayModel.
    """

    assert issubclass(ArrayModel, models.StructModel)
    assert issubclass(ArrayModel, models.ArrayModel)
