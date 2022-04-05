# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numba import types
from numba.core.datamodel import default_manager, models

from numba_dppy.dpnp_ndarray import dpnp_ndarray_Type
from numba_dppy.dpnp_ndarray.models import dpnp_ndarray_Model


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
