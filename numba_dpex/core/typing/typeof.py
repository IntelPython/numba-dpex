# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dpctl import SyclEvent, SyclQueue
from dpctl.tensor import usm_ndarray
from dpnp import ndarray
from numba.extending import typeof_impl
from numba.np import numpy_support

from numba_dpex.kernel_api import AtomicRef, Group, Item, LocalAccessor, NdItem
from numba_dpex.kernel_api.memory_enums import AddressSpace as address_space
from numba_dpex.kernel_api.ranges import NdRange, Range

from ..types.dpctl_types import DpctlSyclEvent, DpctlSyclQueue
from ..types.dpnp_ndarray_type import DpnpNdArray
from ..types.kernel_api.atomic_ref import AtomicRefType
from ..types.kernel_api.index_space_ids import GroupType, ItemType, NdItemType
from ..types.kernel_api.local_accessor import LocalAccessorType
from ..types.kernel_api.ranges import NdRangeType, RangeType
from ..types.usm_ndarray_type import USMNdArray


def _array_typeof_helper(val, array_class_type):
    """Creates a Numba type of the specified ``array_class_type`` for ``val``."""
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))

    try:
        layout = numpy_support.map_layout(val)
    except AttributeError:
        raise ValueError("The layout for the usm_ndarray could not be inferred")

    try:
        # FIXME: Change to readonly = not val.flags.writeable once dpctl is
        # fixed
        readonly = False
    except AttributeError:
        readonly = False

    try:
        usm_type = val.usm_type
    except AttributeError:
        raise ValueError(
            "The usm_type for the usm_ndarray could not be inferred"
        )

    if not val.sycl_queue:
        raise AssertionError

    ty_queue = DpctlSyclQueue(sycl_queue=val.sycl_queue)

    return array_class_type(
        dtype=dtype,
        ndim=val.ndim,
        layout=layout,
        readonly=readonly,
        usm_type=usm_type,
        queue=ty_queue,
        addrspace=address_space.GLOBAL.value,
    )


@typeof_impl.register(usm_ndarray)
def typeof_usm_ndarray(val, c):
    """Registers the type inference implementation function for
    dpctl.tensor.usm_ndarray

    Args:
        val : A Python object that should be an instance of a
        dpctl.tensor.usm_ndarray
        c : Unused argument used to be consistent with Numba API.

    Raises:
        ValueError: If an unsupported dtype encountered or val has
        no ``usm_type`` or sycl_device attribute.

    Returns: The Numba type corresponding to dpctl.tensor.usm_ndarray
    """
    return _array_typeof_helper(val, USMNdArray)


@typeof_impl.register(ndarray)
def typeof_dpnp_ndarray(val, c):
    """Registers the type inference implementation function for dpnp.ndarray.

    Args:
        val : A Python object that should be an instance of a
        dpnp.ndarray
        c : Unused argument used to be consistent with Numba API.

    Raises:
        ValueError: If an unsupported dtype encountered or val has
        no ``usm_type`` or sycl_device attribute.

    Returns: The Numba type corresponding to dpnp.ndarray
    """
    return _array_typeof_helper(val, DpnpNdArray)


@typeof_impl.register(SyclQueue)
def typeof_dpctl_sycl_queue(val, c):
    """Registers the type inference implementation function for a
    dpctl.SyclQueue PyObject.

    Args:
        val : An instance of dpctl.SyclQueue.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.core.types.dpctl_types.DpctlSyclQueue instance.
    """
    return DpctlSyclQueue(val)


@typeof_impl.register(SyclEvent)
def typeof_dpctl_sycl_event(val, c):
    """Registers the type inference implementation function for a
    dpctl.SyclEvent PyObject.

    Args:
        val : An instance of dpctl.SyclEvent.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.core.types.dpctl_types.DpctlSyclEvent instance.
    """
    return DpctlSyclEvent()


@typeof_impl.register(Range)
def typeof_range(val, c):
    """Registers the type inference implementation function for a
    numba_dpex.Range PyObject.

    Args:
        val : An instance of numba_dpex.Range.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.core.types.range_types.RangeType instance.
    """
    return RangeType(val.ndim)


@typeof_impl.register(NdRange)
def typeof_ndrange(val, c):
    """Registers the type inference implementation function for a
    numba_dpex.NdRange PyObject.

    Args:
        val : An instance of numba_dpex.Range.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.core.types.range_types.RangeType instance.
    """
    return NdRangeType(val.global_range.ndim)


@typeof_impl.register(AtomicRef)
def typeof_atomic_ref(val: AtomicRef, ctx) -> AtomicRefType:
    """Returns a ``numba_dpex.experimental.dpctpp_types.AtomicRefType``
    instance for a Python AtomicRef object.

    Args:
        val (AtomicRef): Instance of the AtomicRef type.
        ctx : Numba typing context used for type inference.

    Returns: AtomicRefType object corresponding to the AtomicRef object.

    """
    dtype = typeof_impl(val.ref, ctx)

    return AtomicRefType(
        dtype=dtype,
        memory_order=val.memory_order.value,
        memory_scope=val.memory_scope.value,
        address_space=val.address_space.value,
    )


@typeof_impl.register(Group)
def typeof_group(val: Group, c):
    """Registers the type inference implementation function for a
    numba_dpex.kernel_api.Group PyObject.

    Args:
        val : An instance of numba_dpex.kernel_api.Group.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.experimental.core.types.kernel_api.items.GroupType
        instance.
    """
    return GroupType(val.ndim)


@typeof_impl.register(Item)
def typeof_item(val: Item, c):
    """Registers the type inference implementation function for a
    numba_dpex.kernel_api.Item PyObject.

    Args:
        val : An instance of numba_dpex.kernel_api.Item.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.experimental.core.types.kernel_api.items.ItemType
        instance.
    """
    return ItemType(val.dimensions)


@typeof_impl.register(NdItem)
def typeof_nditem(val: NdItem, c):
    """Registers the type inference implementation function for a
    numba_dpex.kernel_api.NdItem PyObject.

    Args:
        val : An instance of numba_dpex.kernel_api.NdItem.
        c : Unused argument used to be consistent with Numba API.

    Returns: A numba_dpex.experimental.core.types.kernel_api.items.NdItemType
        instance.
    """
    return NdItemType(val.dimensions)


@typeof_impl.register(LocalAccessor)
def typeof_local_accessor(val: LocalAccessor, c) -> LocalAccessorType:
    """Returns a ``numba_dpex.experimental.dpctpp_types.LocalAccessorType``
    instance for a Python LocalAccessor object.
    Args:
        val (LocalAccessor): Instance of the LocalAccessor type.
        c : Numba typing context used for type inference.
    Returns: LocalAccessorType object corresponding to the LocalAccessor object.
    """
    # pylint: disable=protected-access
    return LocalAccessorType(ndim=len(val._shape), dtype=val._dtype)
