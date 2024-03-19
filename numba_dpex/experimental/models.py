# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the Numba data models for the numba_dpex types introduced in the
numba_dpex.experimental module.
"""

from numba.core import types
from numba.core.datamodel import DataModelManager, models
from numba.core.datamodel.models import StructModel
from numba.core.extending import register_model

import numba_dpex.core.datamodel.models as dpex_core_models
from numba_dpex.core.datamodel.models import USMArrayDeviceModel
from numba_dpex.core.types.kernel_api.index_space_ids import (
    GroupType,
    ItemType,
    NdItemType,
)

from ..core.types.kernel_api.atomic_ref import AtomicRefType
from ..core.types.kernel_api.local_accessor import (
    DpctlMDLocalAccessorType,
    LocalAccessorType,
)
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


class EmptyStructModel(StructModel):
    """Data model that does not take space. Intended to be used with types that
    are presented only at typing stage and not represented physically."""

    def __init__(self, dmm, fe_type):
        members = []
        super().__init__(dmm, fe_type, members)


class DpctlMDLocalAccessorModel(StructModel):
    """Data model to represent DpctlMDLocalAccessorType.

    Must be the same structure as
    dpctl/syclinterface/dpctl_sycl_queue_interface.h::MDLocalAccessor.

    Structure intended to be used only on host side of the kernel call.
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("ndim", types.size_t),
            ("dpctl_type_id", types.int32),
            ("dim0", types.size_t),
            ("dim1", types.size_t),
            ("dim2", types.size_t),
        ]
        super().__init__(dmm, fe_type, members)


class LocalAccessorModel(StructModel):
    """Data model for the LocalAccessor type when used in a host-only function."""

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ("shape", types.UniTuple(types.intp, ndim)),
        ]
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
    dmm.register(AtomicRefType, AtomicRefModel)

    # Register the LocalAccessorType type
    dmm.register(LocalAccessorType, USMArrayDeviceModel)

    # Register the GroupType type
    dmm.register(GroupType, EmptyStructModel)

    # Register the ItemType type
    dmm.register(ItemType, EmptyStructModel)

    # Register the NdItemType type
    dmm.register(NdItemType, EmptyStructModel)

    return dmm


exp_dmm = _init_exp_data_model_manager()

# Register any new type that should go into numba.core.datamodel.default_manager
register_model(KernelDispatcherType)(models.OpaqueModel)

# Register the GroupType type
register_model(GroupType)(EmptyStructModel)

# Register the ItemType type
register_model(ItemType)(EmptyStructModel)

# Register the NdItemType type
register_model(NdItemType)(EmptyStructModel)

# Register the MDLocalAccessorType type
register_model(DpctlMDLocalAccessorType)(DpctlMDLocalAccessorModel)

# Register the LocalAccessorType type
register_model(LocalAccessorType)(LocalAccessorModel)
