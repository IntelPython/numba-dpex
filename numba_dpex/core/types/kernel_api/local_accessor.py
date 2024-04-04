# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import cgutils
from numba.core.types import Type, UniTuple, intp
from numba.extending import NativeValue, unbox
from numba.np import numpy_support

from numba_dpex.core.types import USMNdArray
from numba_dpex.kernel_api.memory_enums import AddressSpace


class DpctlMDLocalAccessorType(Type):
    """numba-dpex internal type to represent a dpctl SyclInterface type
    `MDLocalAccessorTy`.
    """

    def __init__(self):
        super().__init__(name="DpctlMDLocalAccessor")


class LocalAccessorType(USMNdArray):
    """numba-dpex internal type to represent a Python object of
    :class:`numba_dpex.experimental.kernel_iface.LocalAccessor`.
    """

    def __init__(self, ndim, dtype):
        try:
            if isinstance(dtype, Type):
                parsed_dtype = dtype
            else:
                parsed_dtype = numpy_support.from_dtype(dtype)
        except NotImplementedError as exc:
            raise ValueError(f"Unsupported array dtype: {dtype}") from exc

        type_name = (
            f"LocalAccessor(dtype={parsed_dtype}, ndim={ndim}, "
            f"address_space={AddressSpace.LOCAL.value})"
        )

        super().__init__(
            ndim=ndim,
            layout="C",
            dtype=parsed_dtype,
            addrspace=AddressSpace.LOCAL.value,
            name=type_name,
        )

    def cast_python_value(self, args):
        """The helper function is not overloaded and using it on the
        LocalAccessorType throws a NotImplementedError.
        """
        raise NotImplementedError


@unbox(LocalAccessorType)
def unbox_local_accessor(typ, obj, c):  # pylint: disable=unused-argument
    """Unboxes a Python LocalAccessor PyObject* into a numba-dpex internal
    representation.

    A LocalAccessor object is represented internally in numba-dpex with the
    same data model as a numpy.ndarray. It is done as a LocalAccessor object
    serves only as a placeholder type when passed to ``call_kernel`` and the
    data buffer should never be accessed inside a host-side compiled function
    such as ``call_kernel``.

    When a LocalAccessor object is passed as an argument to a kernel function
    it uses the USMArrayDeviceModel. Doing so allows numba-dpex to correctly
    generate the kernel signature passing in a pointer in the local address
    space.
    """
    shape = c.pyapi.object_getattr_string(obj, "_shape")
    local_accessor = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    ty_unituple = UniTuple(intp, typ.ndim)
    ll_shape = c.unbox(ty_unituple, shape)
    local_accessor.shape = ll_shape.value

    return NativeValue(
        c.builder.load(local_accessor._getpointer()),
        is_error=ll_shape.is_error,
        cleanup=ll_shape.cleanup,
    )
