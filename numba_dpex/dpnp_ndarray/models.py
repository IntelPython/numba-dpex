# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.datamodel.models import ArrayModel as dpnp_ndarray_Model
from numba.extending import register_model

from numba_dpex.target import spirv_data_model_manager

from .types import dpnp_ndarray_Type

# This tells Numba to use the default Numpy ndarray data layout for
# object of type dpnp.ndarray.
register_model(dpnp_ndarray_Type)(dpnp_ndarray_Model)
spirv_data_model_manager.register(dpnp_ndarray_Type, dpnp_ndarray_Model)
