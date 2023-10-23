# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core.datamodel import models
from numba.core.extending import register_model

from numba_dpex.core.datamodel.models import dpex_data_model_manager as dmm

from .types import KernelDispatcherType

# Register the types and datamodel in the DpexKernelTargetContext
dmm.register(KernelDispatcherType, models.OpaqueModel)

# Register the types and datamodel in the DpexTargetContext
register_model(KernelDispatcherType)(models.OpaqueModel)
