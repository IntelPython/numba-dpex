# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from llvmlite import ir as llvmir
from numba.core import datamodel, types
from numba.core.datamodel.models import OpaqueModel, PrimitiveModel, StructModel

from numba_dpex.core.exceptions import UnreachableError
from numba_dpex.core.types.kernel_api.atomic_ref import AtomicRefType
from numba_dpex.core.types.kernel_api.index_space_ids import (
    GroupType,
    ItemType,
    NdItemType,
)
from numba_dpex.core.types.kernel_api.local_accessor import (
    DpctlMDLocalAccessorType,
    LocalAccessorType,
)
from numba_dpex.kernel_api.memory_enums import AddressSpace as address_space

from ..types import (
    DpctlSyclEvent,
    DpctlSyclQueue,
    DpnpNdArray,
    IntEnumLiteral,
    KernelDispatcherType,
    NdRangeType,
    RangeType,
    USMNdArray,
)


def get_flattened_member_count(ty):
    """Returns the number of fields in an instance of a given StructModel."""

    flattened_member_count = 0
    members = ty._members
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


class GenericPointerModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        adrsp = (
            fe_type.addrspace
            if fe_type.addrspace is not None
            else address_space.GLOBAL.value
        )
        be_type = dmm.lookup(fe_type.dtype).get_data_type().as_pointer(adrsp)
        super(GenericPointerModel, self).__init__(dmm, fe_type, be_type)


class IntEnumLiteralModel(PrimitiveModel):
    """Representation of an object of LiteralIntEnum type using Numba's
    PrimitiveModel that can be represented natively in the target in all
    usage contexts.
    """

    def __init__(self, dmm, fe_type):
        be_type = llvmir.IntType(fe_type.bitwidth)
        super().__init__(dmm, fe_type, be_type)


class USMArrayDeviceModel(StructModel):
    """A data model to represent a usm array type in the LLVM IR generated for a
    device-only kernel function.

    The USMArrayDeviceModel adds an extra address space attribute to the data
    member. The extra attribute is needed when passing usm_ndarray array
    arguments to kernels that are compiled for certain OpenCL GPU devices. Note
    that the address space attribute is applied only to the data member and not
    other members of USMArrayDeviceModel that are pointers. It is done this way
    as other pointer members such as meminfo are not used inside a kernel and
    these members maybe removed from the USMArrayDeviceModel in
    future (refer #929).

    We use separate data models for host (USMArrayHostModel) and device
    (USMArrayDeviceModel) as the address space attribute is only required for
    kernel functions and not needed for functions that are compiled for a host
    memory space.
    """

    # TODO: Evaluate the need to pass meminfo and parent attributes of an array
    #   as kernel params: https://github.com/IntelPython/numba-dpex/issues/929

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ("nitems", types.intp),
            ("itemsize", types.intp),
            (
                "data",
                types.CPointer(fe_type.dtype, addrspace=fe_type.addrspace),
            ),
            ("shape", types.UniTuple(types.intp, ndim)),
            ("strides", types.UniTuple(types.intp, ndim)),
        ]
        super(USMArrayDeviceModel, self).__init__(dmm, fe_type, members)

    @property
    def flattened_field_count(self):
        """
        Return the number of fields in an instance of a USMArrayDeviceModel.
        """
        return get_flattened_member_count(self)


class USMArrayHostModel(StructModel):
    """Data model for the USMNdArray type when used in a host-only function.

    USMArrayHostModel is used by the numba_dpex.types.USMNdArray and
    numba_dpex.types.DpnpNdArray type and abstracts the usmarystruct_t C type
    defined in numba_dpex.core.runtime._usmarraystruct.h.

    The USMArrayDeviceModel differs from numba's ArrayModel by including an
    extra member sycl_queue that maps to _usmarraystruct.sycl_queue pointer. The
    _usmarraystruct.sycl_queue pointer stores the C++ sycl::queue pointer that
    was used to allocate the data for the dpctl.tensor.usm_ndarray or
    dpnp.ndarray represented by an instance of _usmarraystruct.
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
        super(USMArrayHostModel, self).__init__(dmm, fe_type, members)

    @property
    def flattened_field_count(self):
        """Return the number of fields in an instance of a USMArrayHostModel."""
        return get_flattened_member_count(self)


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
                "meminfo",
                types.MemInfoPointer(types.pyobject),
            ),
            (
                "parent",
                types.pyobject,
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
                "meminfo",
                types.MemInfoPointer(types.pyobject),
            ),
            (
                "parent",
                types.pyobject,
            ),
            (
                "event_ref",
                types.CPointer(types.int8),
            ),
        ]
        super(SyclEventModel, self).__init__(dmm, fe_type, members)


class RangeModel(StructModel):
    """The numba_dpex data model for a numba_dpex.kernel_api.Range PyObject."""

    def __init__(self, dmm, fe_type):
        members = [
            ("ndim", types.int64),
            ("dim0", types.int64),
            ("dim1", types.int64),
            ("dim2", types.int64),
        ]
        super(RangeModel, self).__init__(dmm, fe_type, members)

    @property
    def flattened_field_count(self):
        """Return the number of fields in an instance of a RangeModel."""
        return get_flattened_member_count(self)


class NdRangeModel(StructModel):
    """
    The numba_dpex data model for a numba_dpex.kernel_api.NdRange PyObject.
    """

    def __init__(self, dmm, fe_type):
        members = [
            ("ndim", types.int64),
            ("gdim0", types.int64),
            ("gdim1", types.int64),
            ("gdim2", types.int64),
            ("ldim0", types.int64),
            ("ldim1", types.int64),
            ("ldim2", types.int64),
        ]
        super(NdRangeModel, self).__init__(dmm, fe_type, members)

    @property
    def flattened_field_count(self):
        """Return the number of fields in an instance of a NdRangeModel."""
        return get_flattened_member_count(self)


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
    """
    Data model for the LocalAccessor type when used in a host-only function.
    """

    def __init__(self, dmm, fe_type):
        ndim = fe_type.ndim
        members = [
            ("shape", types.UniTuple(types.intp, ndim)),
        ]
        super().__init__(dmm, fe_type, members)


def _init_kernel_data_model_manager() -> datamodel.DataModelManager:
    """Initializes a data model manager used by the SPRIVTarget.

    SPIRV kernel functions for certain types of devices require an explicit
    address space qualifier for pointers. For OpenCL HD Graphics
    devices, defining a kernel function (spir_kernel calling convention) with
    pointer arguments that have no address space qualifier causes a run time
    crash. For this reason, numba-dpex defines two separate data
    models: USMArrayDeviceModel and USMArrayHostModel. When a dpnp.ndarray
    object is passed as an argument to a ``numba_dpex.kernel`` decorated
    function it uses the USMArrayDeviceModel and when passed to a
    ``numba_dpex.dpjit`` decorated function it uses the USMArrayHostModel.
    The difference is due to the fact that inside a ``dpjit`` decorated function
    a dpnp.ndarray object can be passed to any other regular function.

    Returns:
        DataModelManager: A numba-dpex SPIRVTarget-specific data model manager
    """
    dmm = datamodel.default_manager.copy()
    dmm.register(types.CPointer, GenericPointerModel)

    # Register the USMNdArray type to USMArrayDeviceModel in numba_dpex's data
    # model manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(USMNdArray, USMArrayDeviceModel)

    # Register the DpnpNdArray type to USMArrayDeviceModel in numba_dpex's data
    # model manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(DpnpNdArray, USMArrayDeviceModel)

    # Register the DpctlSyclQueue type to SyclQueueModel in numba_dpex's data
    # model manager. The dpex_data_model_manager is used by the DpexKernelTarget
    dmm.register(DpctlSyclQueue, SyclQueueModel)

    dmm.register(IntEnumLiteral, IntEnumLiteralModel)

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


def _init_dpjit_data_model_manager() -> datamodel.DataModelManager:
    # TODO: copy manager
    dmm = datamodel.default_manager

    # Register the USMNdArray type to USMArrayHostModel in numba's default data
    # model manager
    dmm.register(USMNdArray, USMArrayHostModel)

    # Register the DpnpNdArray type to USMArrayHostModel in numba's default data
    # model manager
    dmm.register(DpnpNdArray, USMArrayHostModel)

    # Register the DpctlSyclQueue type
    dmm.register(DpctlSyclQueue, SyclQueueModel)

    # Register the DpctlSyclEvent type
    dmm.register(DpctlSyclEvent, SyclEventModel)

    # Register the RangeType type
    dmm.register(RangeType, RangeModel)

    # Register the NdRangeType type
    dmm.register(NdRangeType, NdRangeModel)

    # Register the GroupType type
    dmm.register(GroupType, EmptyStructModel)

    # Register the ItemType type
    dmm.register(ItemType, EmptyStructModel)

    # Register the NdItemType type
    dmm.register(NdItemType, EmptyStructModel)

    # Register the MDLocalAccessorType type
    dmm.register(DpctlMDLocalAccessorType, DpctlMDLocalAccessorModel)

    # Register the LocalAccessorType type
    dmm.register(LocalAccessorType, LocalAccessorModel)

    # Register the KernelDispatcherType type
    dmm.register(KernelDispatcherType, OpaqueModel)

    return dmm


dpex_data_model_manager = _init_kernel_data_model_manager()
dpjit_data_model_manager = _init_dpjit_data_model_manager()
