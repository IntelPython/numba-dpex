# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import datamodel, types
from numba.core.datamodel.models import PrimitiveModel, StructModel
from numba.core.extending import register_model

from numba_dpex.core.exceptions import UnreachableError
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


class USMArrayModel(StructModel):
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
                types.CPointer(types.pyobject, addrspace=fe_type.addrspace),
            ),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            (
                "data",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            (
                "sycl_queue",
                types.CPointer(types.void, addrspace=fe_type.addrspace),
            ),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super(USMArrayModel, self).__init__(dmm, fe_type, members)


class DpnpNdArrayModel(StructModel):
    """Data model for the DpnpNdArray type.

    The data model for DpnpNdArray is similar to numb's ArrayModel used for
    the numba.types.Array type, with the additional field ``sycl_queue`. The
    `sycl_queue` attribute stores the pointer to the C++ sycl::queue object
    that was used to allocate memory for numba-dpex's native representation
    for an Python object inferred as a DpnpNdArray.
    """

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ("meminfo", types.MemInfoPointer(fe_type.dtype)),
            ("parent", types.pyobject),
            ("nitems", types.intp),
            ("itemsize", types.intp),
            ("data", types.CPointer(fe_type.dtype)),
            ("sycl_queue", types.voidptr),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super(DpnpNdArrayModel, self).__init__(dmm, fe_type, members)

    @property
    def flattened_field_count(self):
        """Return the number of fields in an instance of a DpnpNdArrayModel."""
        flattened_member_count = 0
        members = self._members
        for member in members:
            if isinstance(member, types.UniTuple):
                flattened_member_count += member.count
            elif isinstance(
                member,
                (
                    types.scalars.Integer,
                    types.misc.PyObject,
                    types.misc.RawPointer,
                    types.misc.CPointer,
                    types.misc.MemInfoPointer,
                ),
            ):
                flattened_member_count += 1
            else:
                print(member, type(member))
                raise UnreachableError

        return flattened_member_count


class SyclQueueModel(StructModel):
    """Represents the native data model for a dpctl.SyclQueue PyObject.

    Numba-dpex uses a C struct as defined in
    numba_dpex/core/runtime._queuestruct.h to store the required attributes for
    a ``dpctl.SyclQueue`` Python object.

        - ``queue_ref``: An opaque C pointer to an actual SYCL queue C++ object.
        - ``parent``: A PyObject* that stores a reference back to the original
                      ``dpctl.SyclQueue`` PyObject if the native struct is
                      created by unboxing the PyObject.
    """

    def __init__(self, dmm, fe_type):
        members = [
            (
                "parent",
                types.CPointer(types.int8),
            ),
            (
                "queue_ref",
                types.CPointer(types.int8),
            ),
        ]
        super(SyclQueueModel, self).__init__(dmm, fe_type, members)


def _init_data_model_manager():
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    dmm.register(Array, USMArrayModel)
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
register_model(USMNdArray)(USMArrayModel)
dpex_data_model_manager.register(USMNdArray, USMArrayModel)

# Register the DpnpNdArray type with the Numba ArrayModel
register_model(DpnpNdArray)(DpnpNdArrayModel)
dpex_data_model_manager.register(DpnpNdArray, DpnpNdArrayModel)

# Register the DpctlSyclQueue type
register_model(DpctlSyclQueue)(SyclQueueModel)
dpex_data_model_manager.register(DpctlSyclQueue, SyclQueueModel)
