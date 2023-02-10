# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import datamodel, types
from numba.core.datamodel.models import ArrayModel as DpnpNdArrayModel
from numba.core.datamodel.models import PrimitiveModel, StructModel
from numba.core.extending import register_model

from numba_dpex.core.types import Array, DpnpNdArray, USMNdArray
from numba_dpex.utils import address_space


class GenericPointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        adrsp = (
            fe_type.addrspace
            if fe_type.addrspace is not None
            else address_space.GLOBAL
        )
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)


class ArrayModel(StructModel):
    """A data model to represent a Dpex's array types in LLVM IR.

    Dpex's ArrayModel is based on Numba's ArrayModel for NumPy arrays. The
    dpex model adds an extra address space attribute to all pointer members
    in the array.
    """

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            (
                "meminfo",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            (
                "parent",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            (
                "data",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super(ArrayModel, self).__init__(dmm, fe_type, members)


def _init_data_model_manager():
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    dmm.register(Array, ArrayModel)
    return dmm


dpex_data_model_manager = _init_data_model_manager()

# Register the USMNdArray type with the dpex ArrayModel
register_model(USMNdArray)(ArrayModel)
dpex_data_model_manager.register(USMNdArray, ArrayModel)

# Register the DpnpNdArray type with the dpex ArrayModel
register_model(DpnpNdArray)(DpnpNdArrayModel)
dpex_data_model_manager.register(DpnpNdArray, DpnpNdArrayModel)
