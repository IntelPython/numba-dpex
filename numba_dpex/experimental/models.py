# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Provides the Numba data models for the numba_dpex types introduced in the
numba_dpex.experimental module.
"""

from llvmlite import ir as llvmir
from numba.core import types
from numba.core.datamodel import models
from numba.core.datamodel.models import PrimitiveModel, StructModel
from numba.core.extending import register_model

from numba_dpex.experimental.target import dpex_exp_kernel_target

from .dpcpp_types import AtomicRefType
from .literal_intenum_type import IntEnumLiteral
from .types import KernelDispatcherType


class AtomicRefModel(StructModel):
    """Datamodel for AtomicRefType."""

    def __init__(self, dmm, fe_type):
        members = [
            (
                "ref",
                types.CPointer(fe_type.dtype, addrspace=fe_type.address_space),
            ),
        ]
        super().__init__(dmm, fe_type, members)


class LiteralIntEnumModel(PrimitiveModel):
    """Representation of an object of LiteralIntEnum type using Numba's
    PrimitiveModel that can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, fe_type):
        be_type = llvmir.IntType(fe_type.bitwidth)
        super().__init__(dmm, fe_type, be_type)


# Register the types and datamodel in the DpexKernelTargetContext
exp_dmm = dpex_exp_kernel_target.target_context.data_model_manager

exp_dmm.register(KernelDispatcherType, models.OpaqueModel)
exp_dmm.register(AtomicRefType, AtomicRefModel)
exp_dmm.register(IntEnumLiteral, LiteralIntEnumModel)

# Register the types and datamodel in the DpexTargetContext
register_model(KernelDispatcherType)(models.OpaqueModel)
