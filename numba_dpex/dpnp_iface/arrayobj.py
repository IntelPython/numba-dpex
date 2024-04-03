# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba import errors, types
from numba.core.types import scalars
from numba.core.types.containers import UniTuple
from numba.core.typing.npydecl import parse_dtype as _ty_parse_dtype
from numba.core.typing.npydecl import parse_shape as _ty_parse_shape
from numba.extending import overload
from numba.np.numpy_support import is_nonelike

from numba_dpex.core.types import DpnpNdArray

from ._intrinsic import (
    impl_dpnp_empty,
    impl_dpnp_empty_like,
    impl_dpnp_full,
    impl_dpnp_full_like,
    impl_dpnp_ones,
    impl_dpnp_ones_like,
    impl_dpnp_zeros,
    impl_dpnp_zeros_like,
)

# can't import name because of the circular import
DPEX_TARGET_NAME = "dpex"

# =========================================================================
#               Helps to parse dpnp constructor arguments
# =========================================================================


def _parse_dim(x1):
    if hasattr(x1, "ndim") and x1.ndim:
        return x1.ndim
    elif isinstance(x1, scalars.Integer):
        r = 1
        return r
    elif isinstance(x1, UniTuple):
        r = len(x1)
        return r
    else:
        return 0


def _parse_dtype(dtype):
    """Resolve dtype parameter.

    Resolves the dtype parameter based on the given value
    or the dtype of the given array.

    Args:
        dtype (numba.core.types.functions.NumberClass): Numba type
            class for number classes (e.g. "np.float64").
        data (numba.core.types.npytypes.Array, optional): Numba type
            class for nd-arrays. Defaults to None.

    Returns:
        numba.core.types.functions.NumberClass: Resolved numba type
            class for number classes.
    """

    _dtype = None
    if not is_nonelike(dtype):
        _dtype = _ty_parse_dtype(dtype)
    return _dtype


def _parse_layout(layout):
    if isinstance(layout, types.StringLiteral):
        layout_type_str = layout.literal_value
        if layout_type_str not in ["C", "F", "A"]:
            msg = f"Invalid layout specified: '{layout_type_str}'"
            raise errors.NumbaValueError(msg)
        return layout_type_str
    elif isinstance(layout, str):
        if layout not in ["C", "F", "A"]:
            msg = f"Invalid layout specified: '{layout}'"
            raise errors.NumbaValueError(msg)
        return layout
    else:
        raise TypeError(
            "The parameter 'layout' is neither of "
            + "'str' nor 'types.StringLiteral'"
        )


def _parse_usm_type(usm_type):
    """Parse usm_type parameter.

    Resolves the usm_type parameter based on the type
    of the parameter.

    Args:
        usm_type (str, numba.core.types.misc.StringLiteral):
            The type class for the string to specify the usm_type.

    Raises:
        errors.NumbaValueError: If an invalid usm_type is specified.
        TypeError: If the parameter is neither a 'str'
                    nor a 'types.StringLiteral'.

    Returns:
        str: The stringized usm_type.
    """

    if isinstance(usm_type, types.StringLiteral):
        usm_type_str = usm_type.literal_value
        if usm_type_str not in ["shared", "device", "host"]:
            msg = f"Invalid usm_type specified: '{usm_type_str}'"
            raise errors.NumbaValueError(msg)
        return usm_type_str
    elif isinstance(usm_type, str):
        if usm_type not in ["shared", "device", "host"]:
            msg = f"Invalid usm_type specified: '{usm_type}'"
            raise errors.NumbaValueError(msg)
        return usm_type
    else:
        raise TypeError(
            "The parameter 'usm_type' is neither of "
            + "'str' nor 'types.StringLiteral'"
        )


def _parse_device_filter_string(device):
    """Parse the device type parameter.

    Returns the device filter string,
    if it is a string literal.

    Args:
        device (str, numba.core.types.misc.StringLiteral):
            The type class for the string to specify the device.

    Raises:
        TypeError: If the parameter is neither a 'str'
                    nor a 'types.StringLiteral'.

    Returns:
        str: The stringized device.
    """

    if isinstance(device, types.StringLiteral):
        device_filter_str = device.literal_value
        return device_filter_str
    elif isinstance(device, str):
        return device
    elif device is None or isinstance(device, types.NoneType):
        return None
    else:
        raise TypeError(
            "The parameter 'device' is neither of "
            + "'str', 'types.StringLiteral' nor 'None'"
        )


# =========================================================================
#                       Dpnp array constructor overloads
# =========================================================================


@overload(dpnp.empty, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_empty(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.empty() inside
    a dpjit function.

    Args:
        shape (numba.core.types.containers.UniTuple or
            numba.core.types.scalars.IntegerLiteral): Dimensions
            of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.empty().

    Returns:
        function: Local function `impl_dpnp_empty()`.
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _layout = _parse_layout(order)
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    if _ndim:
        ret_ty = DpnpNdArray(
            ndim=_ndim,
            layout=_layout,
            dtype=_dtype,
            usm_type=_usm_type,
            device=_device,
            queue=sycl_queue,
        )

        if ret_ty:

            def impl(
                shape,
                dtype=None,
                order="C",
                # like=None, # noqa: E800 see issue https://github.com/IntelPython/numba-dpex/issues/998
                device=None,
                usm_type="device",
                sycl_queue=None,
            ):
                return impl_dpnp_empty(
                    shape,
                    _dtype,
                    order,
                    # like, # noqa: E800 see issue https://github.com/IntelPython/numba-dpex/issues/998
                    _device,
                    _usm_type,
                    sycl_queue,
                    ret_ty,
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.empty({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.zeros, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_zeros(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.zeros() inside
    a dpjit function.

    Args:
        shape (numba.core.types.containers.UniTuple or
            numba.core.types.scalars.IntegerLiteral): Dimensions
            of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.zeros().

    Returns:
        function: Local function `impl_dpnp_zeros()`.
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _layout = _parse_layout(order)
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    if _ndim:
        ret_ty = DpnpNdArray(
            ndim=_ndim,
            layout=_layout,
            dtype=_dtype,
            usm_type=_usm_type,
            device=_device,
            queue=sycl_queue,
        )
        if ret_ty:

            def impl(
                shape,
                dtype=None,
                order="C",
                device=None,
                usm_type="device",
                sycl_queue=None,
            ):
                return impl_dpnp_zeros(
                    shape,
                    _dtype,
                    order,
                    _device,
                    _usm_type,
                    sycl_queue,
                    ret_ty,
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.zeros({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.ones, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_ones(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.ones() inside
    a dpjit function.

    Args:
        shape (numba.core.types.containers.UniTuple or
            numba.core.types.scalars.IntegerLiteral): Dimensions
            of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.ones().

    Returns:
        function: Local function `impl_dpnp_ones()`.
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _layout = _parse_layout(order)
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    if _ndim:
        ret_ty = DpnpNdArray(
            ndim=_ndim,
            layout=_layout,
            dtype=_dtype,
            usm_type=_usm_type,
            device=_device,
            queue=sycl_queue,
        )
        if ret_ty:

            def impl(
                shape,
                dtype=None,
                order="C",
                device=None,
                usm_type="device",
                sycl_queue=None,
            ):
                return impl_dpnp_ones(
                    shape,
                    _dtype,
                    order,
                    _device,
                    _usm_type,
                    sycl_queue,
                    ret_ty,
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.ones({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.full, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_full(
    shape,
    fill_value,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.full() inside
    a dpjit function.

    Args:
        shape (numba.core.types.containers.UniTuple or
            numba.core.types.scalars.IntegerLiteral): Dimensions
            of the array to be created.
        fill_value (numba.core.types.scalars): One of the
            numba.core.types.scalar types for the value to
            be filled.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        like (numba.core.types.npytypes.Array, optional): A type for
            reference object to allow the creation of arrays which are not
            `NumPy` arrays. If an array-like passed in as `like` supports the
            `__array_function__` protocol, the result will be defined by it.
            In this case, it ensures the creation of an array object
            compatible with that passed in via this argument.
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.full().

    Returns:
        function: Local function `impl_dpnp_full()`.
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype) if dtype is not None else fill_value
    _layout = _parse_layout(order)
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    if _ndim:
        ret_ty = DpnpNdArray(
            ndim=_ndim,
            layout=_layout,
            dtype=_dtype,
            usm_type=_usm_type,
            device=_device,
            queue=sycl_queue,
        )
        if ret_ty:

            def impl(
                shape,
                fill_value,
                dtype=None,
                order="C",
                like=None,
                device=None,
                usm_type=None,
                sycl_queue=None,
            ):
                return impl_dpnp_full(
                    shape,
                    fill_value,
                    _dtype,
                    order,
                    like,
                    _device,
                    _usm_type,
                    sycl_queue,
                    ret_ty,
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.full({shape}, {fill_value}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.empty_like, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_empty_like(
    x1,
    dtype=None,
    order="C",
    subok=False,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from uninitialized USM allocation.

    This is an overloaded function implementation for dpnp.empty_like().

    Args:
        x1 (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        subok ('numba.core.types.scalars.BooleanLiteral', optional): A
            boolean literal type for the `subok` parameter defined in
            NumPy. If True, then the newly created array will use the
            sub-class type of prototype, otherwise it will be a
            base-class array. Defaults to False.
        shape (numba.core.types.containers.UniTuple, optional): The shape
            to override the shape of the given array. Not supported.
            Default: `None`
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If couldn't parse input types to dpnp.empty_like().
        errors.TypingError: If shape is provided.
        errors.TypingError: If `x1` is not an instance of DpnpNdArray

    Returns:
        function: Local function `impl_dpnp_empty_like()`.
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.empty_like() function."
        )

    if not isinstance(x1, DpnpNdArray):
        raise errors.TypingError(
            "Only objects of dpnp.dpnp_array type are supported as "
            "input array ``x1``."
        )

    _ndim = _parse_dim(x1)
    _dtype = x1.dtype if isinstance(x1, types.Array) else _parse_dtype(dtype)
    _layout = x1.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    # If a sycl_queue or device argument was not explicitly provided get the
    # queue from the array (x1) argument.
    _queue = sycl_queue
    if _queue is None and _device is None:
        _queue = x1.queue

    ret_ty = DpnpNdArray(
        ndim=_ndim,
        layout=_layout,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=_queue,
    )

    if ret_ty:

        def impl(
            x1,
            dtype=None,
            order="C",
            subok=False,
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_empty_like(
                x1,
                _dtype,
                _layout,
                subok,
                shape,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x1}, {dtype}, ...)."
        )


@overload(dpnp.zeros_like, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_zeros_like(
    x1,
    dtype=None,
    order="C",
    subok=None,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from USM allocation initialized with zeros.

    This is an overloaded function implementation for dpnp.zeros_like().

    Args:
        x1 (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        subok ('numba.core.types.scalars.BooleanLiteral', optional): A
            boolean literal type for the `subok` parameter defined in
            NumPy. If True, then the newly created array will use the
            sub-class type of prototype, otherwise it will be a
            base-class array. Defaults to False.
        shape (numba.core.types.containers.UniTuple, optional): The shape
            to override the shape of the given array. Not supported.
            Default: `None`
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If couldn't parse input types to dpnp.zeros_like().
        errors.TypingError: If shape is provided.
        errors.TypingError: If `x1` is not an instance of DpnpNdArray

    Returns:
        function: Local function `impl_dpnp_zeros_like()`.
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.zeros_like() function."
        )

    if not isinstance(x1, DpnpNdArray):
        raise errors.TypingError(
            "Only objects of dpnp.dpnp_array type are supported as "
            "input array ``x1``."
        )

    _ndim = _parse_dim(x1)
    _dtype = x1.dtype if isinstance(x1, types.Array) else _parse_dtype(dtype)
    _layout = x1.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    # If a sycl_queue or device argument was not explicitly provided get the
    # queue from the array (x1) argument.
    _queue = sycl_queue
    if _queue is None and _device is None:
        _queue = x1.queue

    ret_ty = DpnpNdArray(
        ndim=_ndim,
        layout=_layout,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=_queue,
    )
    if ret_ty:

        def impl(
            x1,
            dtype=None,
            order="C",
            subok=None,
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_zeros_like(
                x1,
                _dtype,
                _layout,
                subok,
                shape,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x1}, {dtype}, ...)."
        )


@overload(dpnp.ones_like, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_ones_like(
    x1,
    dtype=None,
    order="C",
    subok=None,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from USM allocation initialized with ones.

    This is an overloaded function implementation for dpnp.ones_like().

    Args:
        x1 (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        subok ('numba.core.types.scalars.BooleanLiteral', optional): A
            boolean literal type for the `subok` parameter defined in
            NumPy. If True, then the newly created array will use the
            sub-class type of prototype, otherwise it will be a
            base-class array. Defaults to False.
        shape (numba.core.types.containers.UniTuple, optional): The shape
            to override the shape of the given array. Not supported.
            Default: `None`
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If couldn't parse input types to dpnp.ones_like().
        errors.TypingError: If shape is provided.
        errors.TypingError: If `x1` is not an instance of DpnpNdArray

    Returns:
        function: Local function `impl_dpnp_ones_like()`.
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.ones_like() function."
        )

    if not isinstance(x1, DpnpNdArray):
        raise errors.TypingError(
            "Only objects of dpnp.dpnp_array type are supported as "
            "input array ``x1``."
        )

    _ndim = _parse_dim(x1)
    _dtype = x1.dtype if isinstance(x1, types.Array) else _parse_dtype(dtype)
    _layout = x1.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    # If a sycl_queue or device argument was not explicitly provided get the
    # queue from the array (x1) argument.
    _queue = sycl_queue
    if _queue is None and _device is None:
        _queue = x1.queue

    ret_ty = DpnpNdArray(
        ndim=_ndim,
        layout=_layout,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=_queue,
    )

    if ret_ty:

        def impl(
            x1,
            dtype=None,
            order="C",
            subok=None,
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_ones_like(
                x1,
                _dtype,
                _layout,
                subok,
                shape,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x1}, {dtype}, ...)."
        )


@overload(dpnp.full_like, prefer_literal=True, target=DPEX_TARGET_NAME)
def ol_dpnp_full_like(
    x1,
    fill_value,
    dtype=None,
    order="C",
    subok=None,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from USM allocation initialized with values
    specified by the `fill_value`.

    This is an overloaded function implementation for dpnp.full_like().

    Args:
        x1 (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        fill_value (numba.core.types.scalars): One of the
            numba.core.types.scalar types for the value to
            be filled.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None.
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C".
        subok ('numba.core.types.scalars.BooleanLiteral', optional): A
            boolean literal type for the `subok` parameter defined in
            NumPy. If True, then the newly created array will use the
            sub-class type of prototype, otherwise it will be a
            base-class array. Defaults to False.
        shape (numba.core.types.containers.UniTuple, optional): The shape
            to override the shape of the given array. Not supported.
            Default: `None`
        device (numba.core.types.misc.StringLiteral, optional): array API
            concept of device where the output array is created. `device`
            can be `None`, a oneAPI filter selector string, an instance of
            :class:`dpctl.SyclDevice` corresponding to a non-partitioned
            SYCL device, an instance of :class:`dpctl.SyclQueue`, or a
            `Device` object returnedby`dpctl.tensor.usm_array.device`.
            Default: `None`.
        usm_type (numba.core.types.misc.StringLiteral or str, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        sycl_queue (:class:`numba_dpex.core.types.dpctl_types.DpctlSyclQueue`,
            optional): The SYCL queue to use for output array allocation and
            copying. sycl_queue and device are exclusive keywords, i.e. use
            one or another. If both are specified, a TypeError is raised. If
            both are None, a cached queue targeting default-selected device
            is used for allocation and copying. Default: `None`.

    Raises:
        errors.TypingError: If both `device` and `sycl_queue` are provided.
        errors.TypingError: If couldn't parse input types to dpnp.full_like().
        errors.TypingError: If shape is provided.
        errors.TypingError: If `x1` is not an instance of DpnpNdArray

    Returns:
        function: Local function `impl_dpnp_full_like()`.
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.full_like() function."
        )

    _ndim = _parse_dim(x1)
    _dtype = (
        x1.dtype
        if isinstance(x1, types.Array)
        else (_parse_dtype(dtype) if dtype is not None else fill_value)
    )
    _layout = x1.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type else "device"
    _device = _parse_device_filter_string(device) if device else None

    # If a sycl_queue or device argument was not explicitly provided get the
    # queue from the array (x1) argument.
    _queue = sycl_queue
    if _queue is None and _device is None:
        _queue = x1.queue

    ret_ty = DpnpNdArray(
        ndim=_ndim,
        layout=_layout,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=_queue,
    )

    if ret_ty:

        def impl(
            x1,
            fill_value,
            dtype=None,
            order="C",
            subok=None,
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_full_like(
                x1,
                fill_value,
                _dtype,
                _layout,
                subok,
                shape,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.full_like({x1}, {fill_value}, {dtype}, ...)."
        )
