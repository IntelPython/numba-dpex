# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import datamodel, types
from numba.core.datamodel.models import ArrayModel as DpnpNdArrayModel
from numba.core.datamodel.models import OpaqueModel, PrimitiveModel, StructModel
from numba.core.extending import register_model

from numba_dpex.utils import address_space

from ..types import Array, DpctlSyclQueue, DpnpNdArray, USMNdArray


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

# XXX A kernel function has the spir_kernel ABI and requires pointers to have an
# address space attribute. For this reason, the UsmNdArray type uses dpex's
# ArrayModel where the pointers are address space casted to have a SYCL-specific
# address space value. The DpnpNdArray type can be used inside djit functions
# as host function calls arguments, such as dpnp library calls. The DpnpNdArray
# needs to use Numba's array model as its data model. Thus, from a Numba typing
# perspective dpnp.ndarrays cannot be directly passed to a kernel. To get
# around the limitation, the DpexKernelTypingContext does not resolve the type
# of dpnp.array args to a kernel as DpnpNdArray type objects, but uses the
# ``to_usm_ndarray`` utility function to convert them into a UsmNdArray type
# object.

# Register the USMNdArray type with the dpex ArrayModel
register_model(USMNdArray)(ArrayModel)
dpex_data_model_manager.register(USMNdArray, ArrayModel)

# Register the DpnpNdArray type with the Numba ArrayModel
register_model(DpnpNdArray)(DpnpNdArrayModel)
dpex_data_model_manager.register(DpnpNdArray, DpnpNdArrayModel)

# Register the DpctlSyclQueue type with Numba's OpaqueModel
register_model(DpctlSyclQueue)(OpaqueModel)
dpex_data_model_manager.register(DpctlSyclQueue, OpaqueModel)
