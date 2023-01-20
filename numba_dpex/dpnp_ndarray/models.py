# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.datamodel.models import ArrayModel as dpnp_ndarray_Model
from numba.extending import register_model

from numba_dpex.core.datamodel.models import dpex_data_model_manager

from ..core.types.dpnp_ndarray_type import DpnpNdarray

# This tells Numba to use the default Numpy ndarray data layout for
# object of type dpnp.ndarray.
register_model(DpnpNdarray)(dpnp_ndarray_Model)
dpex_data_model_manager.register(DpnpNdarray, dpnp_ndarray_Model)
