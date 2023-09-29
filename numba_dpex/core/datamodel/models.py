# SPDX-FileCopyrightText: 2022 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import datamodel, types
from numba.core.datamodel.models import PrimitiveModel, StructModel
from numba.core.extending import register_model

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.utils import address_space

from ..types import (
    Array,
    DpctlSyclEvent,
    DpctlSyclQueue,
    DpnpNdArray,
    USMNdArray,
)


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

    @property
    def flattened_field_count(self):
        """Return the number of fields in an instance of a USMArrayModel."""
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
                raise UnreachableError

        return flattened_member_count


class DpnpNdArrayModel(StructModel):
    """Data model for the DpnpNdArray type.

    DpnpNdArrayModel is used by the numba_dpex.types.DpnpNdArray type and
    abstracts the usmarystruct_t C type defined in
    numba_dpex.core.runtime._usmarraystruct.h.

    The DpnpNdArrayModel differs from numba's ArrayModel by including an extra
    member sycl_queue that maps to _usmarraystruct.sycl_queue pointer. The
    _usmarraystruct.sycl_queue pointer stores the C++ sycl::queue pointer that
    was used to allocate the data for the dpnp.ndarray represented by an
    instance of _usmarraystruct.
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


class SyclEventModel(StructModel):
    """Represents the native data model for a dpctl.SyclEvent PyObject.

    Numba-dpex uses a C struct as defined in
    numba_dpex/core/runtime._eventstruct.h to store the required attributes for
    a ``dpctl.SyclEvent`` Python object.

        - ``event_ref``: An opaque C pointer to an actual SYCL event C++ object.
        - ``parent``: A PyObject* that stores a reference back to the original
                      ``dpctl.SyclEvent`` PyObject if the native struct is
                      created by unboxing the PyObject.
    """

    def __init__(self, dmm, fe_type):
        members = [
            (
                "parent",
                types.CPointer(types.int8),
            ),
            (
                "event_ref",
                types.CPointer(types.int8),
            ),
        ]
        super(SyclEventModel, self).__init__(dmm, fe_type, members)


def _init_data_model_manager() -> datamodel.DataModelManager:
    """Initializes a DpexKernelTarget-specific data model manager.

    SPIRV kernel functions for certain types of devices require an explicit
    address space qualifier for pointers. For OpenCL HD Graphics
    devices, defining a kernel function (spir_kernel calling convention) with
    pointer arguments that have no address space qualifier causes a run time
    crash. For this reason, numba-dpex defines two separate data
    models: USMArrayModel and DpnpNdArrayModel. When a dpnp.ndarray object is
    passed as an argument to a ``numba_dpex.kernel`` decorated function it uses
    the USMArrayModel and when passed to a ``numba_dpex.dpjit`` decorated
    function it uses the DpnpNdArrayModel. The difference is due to the fact
    that inside a ``dpjit`` decorated function a dpnp.ndarray object can be
    passed to any other regular function.

    Returns:
        DataModelManager: A numba-dpex DpexKernelTarget-specific data model
        manager
    """
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)
    dmm.register(Array, USMArrayModel)

    # Register the USMNdArray type to USMArrayModel in numba_dpex's data model
    # manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(USMNdArray, USMArrayModel)

    # Register the DpnpNdArray type to USMArrayModel in numba_dpex's data model
    # manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(DpnpNdArray, USMArrayModel)

    # Register the DpctlSyclQueue type to SyclQueueModel in numba_dpex's data
    # model manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(DpctlSyclQueue, SyclQueueModel)

    return dmm


dpex_data_model_manager = _init_data_model_manager()


# Register the USMNdArray type to USMArrayModel in numba's default data model
# manager
register_model(USMNdArray)(USMArrayModel)

# Register the DpnpNdArray type to DpnpNdArrayModel in numba's default data
# model manager
register_model(DpnpNdArray)(DpnpNdArrayModel)

# Register the DpctlSyclQueue type
register_model(DpctlSyclQueue)(SyclQueueModel)

# Register the DpctlSyclEvent type
register_model(DpctlSyclEvent)(SyclEventModel)
