# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides Numba datamodel for the numba_dpex types introduced in the
numba_dpex.experimental module.
"""

from numba.core.datamodel import models
from numba.core.extending import register_model

from numba_dpex.core.datamodel.models import dpex_data_model_manager as dmm

from .types import KernelDispatcherType

# Register the types and datamodel in the DpexKernelTargetContext
dmm.register(KernelDispatcherType, models.OpaqueModel)

# Register the types and datamodel in the DpexTargetContext
register_model(KernelDispatcherType)(models.OpaqueModel)
