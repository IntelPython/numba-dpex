# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba import types
from numba.core.datamodel import default_manager, models

from numba_dpex.dpnp_ndarray import dpnp_ndarray_Type
from numba_dpex.dpnp_ndarray.models import dpnp_ndarray_Model


def test_model_for_dpnp_ndarray_Type():
    """Test that model is registered for dpnp_ndarray_Type instances.

    The model for dpnp_ndarray_Type is dpnp_ndarray_Model.

    """

    model = default_manager.lookup(dpnp_ndarray_Type(types.float64, 1, "C"))
    assert isinstance(model, dpnp_ndarray_Model)


def test_dpnp_ndarray_Model():
    """Test for dpnp_ndarray_Model.

    It is a subclass of models.StructModel and models.ArrayModel.
    """

    assert issubclass(dpnp_ndarray_Model, models.StructModel)
    assert issubclass(dpnp_ndarray_Model, models.ArrayModel)
