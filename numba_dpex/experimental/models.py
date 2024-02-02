# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the Numba data models for the numba_dpex types introduced in the
numba_dpex.experimental module.
"""

from llvmlite import ir as llvmir
from numba.core import types
from numba.core.datamodel import DataModelManager, models
from numba.core.datamodel.models import PrimitiveModel, StructModel
from numba.core.extending import register_model

import numba_dpex.core.datamodel.models as dpex_core_models
from numba_dpex.experimental.core.types.kernel_api.items import (
    ItemType,
    NdItemType,
)

from .dpcpp_types import AtomicRefType
from .literal_intenum_type import IntEnumLiteral
from .types import KernelDispatcherType


class AtomicRefModel(StructModel):
    """Data model for AtomicRefType."""

    def __init__(self, dmm, fe_type):
        members = [
            (
                "ref",
                types.CPointer(fe_type.dtype, addrspace=fe_type.address_space),
            ),
        ]
        super().__init__(dmm, fe_type, members)


class IntEnumLiteralModel(PrimitiveModel):
    """Representation of an object of LiteralIntEnum type using Numba's
    PrimitiveModel that can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, fe_type):
        be_type = llvmir.IntType(fe_type.bitwidth)
        super().__init__(dmm, fe_type, be_type)


class EmptyStructModel(StructModel):
    """Data model that does not take space. Intended to be used with types that
    are presented only at typing stage and not represented physically."""

    def __init__(self, dmm, fe_type):
        members = []
        super().__init__(dmm, fe_type, members)


def _init_exp_data_model_manager() -> DataModelManager:
    """Initializes a DpexExpKernelTarget-specific data model manager.

    Extends the DpexKernelTargetContext's datamodel manager with all
    experimental types that are getting added to the kernel API.

    Returns:
        DataModelManager: A numba-dpex DpexExpKernelTarget-specific data model
        manager
    """

    dmm = dpex_core_models.dpex_data_model_manager.copy()

    # Register the types and data model in the DpexExpTargetContext
    dmm.register(IntEnumLiteral, IntEnumLiteralModel)
    dmm.register(AtomicRefType, AtomicRefModel)

    # Register the ItemType type
    dmm.register(ItemType, EmptyStructModel)

    # Register the NdItemType type
    dmm.register(NdItemType, EmptyStructModel)

    return dmm


exp_dmm = _init_exp_data_model_manager()

# Register any new type that should go into numba.core.datamodel.default_manager
register_model(KernelDispatcherType)(models.OpaqueModel)

# Register the ItemType type
register_model(ItemType)(EmptyStructModel)

# Register the NdItemType type
register_model(NdItemType)(EmptyStructModel)
