# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba import errors, types
from numba.core.typing.npydecl import parse_dtype as _ty_parse_dtype
from numba.core.typing.npydecl import parse_shape as _ty_parse_shape
from numba.extending import overload, overload_classmethod
from numba.np.numpy_support import is_nonelike

from numba_dpex.core.types import DpnpNdArray

from ._intrinsic import (
    impl_dpnp_empty,
    impl_dpnp_empty_like,
    impl_dpnp_ones,
    impl_dpnp_ones_like,
    impl_dpnp_zeros,
    impl_dpnp_zeros_like,
    intrin_usm_alloc,
)

# =========================================================================
#               Helps to parse dpnp constructor arguments
# =========================================================================


def _parse_dtype(dtype, data=None):
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
    if data and isinstance(data, types.Array):
        _dtype = data.dtype
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
                    nor a 'types.StringLiteral'

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
                    nor a 'types.StringLiteral'

    Returns:
        str: The stringized device.
    """

    if isinstance(device, types.StringLiteral):
        device_filter_str = device.literal_value
        return device_filter_str
    elif isinstance(device, str):
        return device
    else:
        raise TypeError(
            "The parameter 'device' is neither of "
            + "'str' nor 'types.StringLiteral'"
        )


def build_dpnp_ndarray(
    ndim,
    layout="C",
    dtype=None,
    usm_type="device",
    device="unknown",
    queue=None,
):
    """Constructs `DpnpNdArray` from the parameters provided.

    Args:
        ndim (int): The dimension of the array.
        layout ("C", or F"): memory layout for the array. Default: "C"
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        usm_type (numba.core.types.misc.StringLiteral, optional):
            The type of SYCL USM allocation for the output array.
            Allowed values are "device"|"shared"|"host".
            Default: `"device"`.
        device (optional): array API concept of device where the
            output array is created. `device` can be `None`, a oneAPI
            filter selector string, an instance of :class:`dpctl.SyclDevice`
            corresponding to a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a `Device` object returnedby
            `dpctl.tensor.usm_array.device`. Default: `"unknwon"`.
        queue (:class:`dpctl.SyclQueue`, optional): Not supported.
            Default: `None`.

    Raises:
        errors.TypingError: If `sycl_queue` is provided for some reason.

    Returns:
        DpnpNdArray: The Numba type to represent an dpnp.ndarray.
            The type has the same structure as USMNdArray used to
            represent dpctl.tensor.usm_ndarray.
    """
    if queue and not isinstance(queue, types.misc.Omitted):
        raise errors.TypingError(
            "The sycl_queue keyword is not yet supported by "
            "dpnp.empty(), dpnp.zeros(), dpnp.ones(), dpnp.empty_like(), "
            "dpnp.zeros_like() and dpnp.ones_like() inside "
            "a dpjit decorated function."
        )

    # If a dtype value was passed in, then try to convert it to the
    # corresponding Numba type. If None was passed, the default, then pass None
    # to the DpnpNdArray constructor. The default dtype will be derived based
    # on the behavior defined in dpctl.tensor.usm_ndarray.

    ret_ty = DpnpNdArray(
        ndim=ndim, layout=layout, dtype=dtype, usm_type=usm_type, device=device
    )

    return ret_ty


# =========================================================================
#                       Dpnp array constructor overloads
# =========================================================================


@overload_classmethod(DpnpNdArray, "_usm_allocate")
def _ol_array_allocate(cls, allocsize, usm_type, device):
    """Implements an allocator for dpnp.ndarrays."""

    def impl(cls, allocsize, usm_type, device):
        return intrin_usm_alloc(allocsize, usm_type, device)

    return impl


@overload(dpnp.empty, prefer_literal=True)
def ol_dpnp_empty(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.empty() inside
    a jit function.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.empty().

    Returns:
        function: Local function `impl_dpnp_empty()`
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _layout = _parse_layout(order)
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    if _ndim:
        ret_ty = build_dpnp_ndarray(
            _ndim,
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
                return impl_dpnp_empty(
                    shape, _dtype, order, _device, _usm_type, sycl_queue, ret_ty
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.empty({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.zeros, prefer_literal=True)
def ol_dpnp_zeros(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.zeros() inside
    a jit function.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.zeros().

    Returns:
        function: Local function `impl_dpnp_zeros()`
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    if _ndim:
        ret_ty = build_dpnp_ndarray(
            _ndim,
            layout=order,
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
                    shape, _dtype, order, _device, _usm_type, sycl_queue, ret_ty
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.zeros({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.ones, prefer_literal=True)
def ol_dpnp_ones(
    shape,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """Implementation of an overload to support dpnp.ones() inside
    a jit function.

    Args:
        shape (tuple): Dimensions of the array to be created.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If rank of the ndarray couldn't be inferred.
        errors.TypingError: If couldn't parse input types to dpnp.ones().

    Returns:
        function: Local function `impl_dpnp_ones()`
    """

    _ndim = _ty_parse_shape(shape)
    _dtype = _parse_dtype(dtype)
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    if _ndim:
        ret_ty = build_dpnp_ndarray(
            _ndim,
            layout=order,
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
                    shape, _dtype, order, _device, _usm_type, sycl_queue, ret_ty
                )

            return impl
        else:
            raise errors.TypingError(
                "Cannot parse input types to "
                + f"function dpnp.ones({shape}, {dtype}, ...)."
            )
    else:
        raise errors.TypingError("Could not infer the rank of the ndarray.")


@overload(dpnp.empty_like, prefer_literal=True)
def ol_dpnp_empty_like(
    x,
    dtype=None,
    order="C",
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from uninitialized USM allocation.

    This is an overloaded function implementation for dpnp.empty_like().

    Args:
        x (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If couldn't parse input types to dpnp.empty_like().
        errors.TypingError: If shape is provided.

    Returns:
        function: Local function `impl_dpnp_empty_like()`
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.empty_like() function."
        )
    _ndim = x.ndim if hasattr(x, "ndim") and x.ndim is not None else 0
    _dtype = _parse_dtype(dtype, data=x)
    _order = x.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    ret_ty = build_dpnp_ndarray(
        _ndim,
        layout=_order,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=sycl_queue,
    )
    if ret_ty:

        def impl(
            x,
            dtype=None,
            order="C",
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_empty_like(
                x,
                _dtype,
                _order,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x}, {dtype}, ...)."
        )


@overload(dpnp.zeros_like, prefer_literal=True)
def ol_dpnp_zeros_like(
    x,
    dtype=None,
    order="C",
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from USM allocation initialized with zeros.

    This is an overloaded function implementation for dpnp.zeros_like().

    Args:
        x (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If couldn't parse input types to dpnp.zeros_like().
        errors.TypingError: If shape is provided.

    Returns:
        function: Local function `impl_dpnp_zeros_like()`
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.zeros_like() function."
        )
    _ndim = x.ndim if hasattr(x, "ndim") and x.ndim is not None else 0
    _dtype = _parse_dtype(dtype, data=x)
    _order = x.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    ret_ty = build_dpnp_ndarray(
        _ndim,
        layout=_order,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=sycl_queue,
    )
    if ret_ty:

        def impl(
            x,
            dtype=None,
            order="C",
            shape=None,
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_zeros_like(
                x,
                _dtype,
                _order,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x}, {dtype}, ...)."
        )


@overload(dpnp.ones_like, prefer_literal=True)
def ol_dpnp_ones_like(
    x,
    dtype=None,
    order="C",
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """Creates `usm_ndarray` from USM allocation initialized with ones.

    This is an overloaded function implementation for dpnp.ones_like().

    Args:
        x (numba.core.types.npytypes.Array): Input array from which to
            derive the output array shape.
        dtype (numba.core.types.functions.NumberClass, optional):
            Data type of the array. Can be typestring, a `numpy.dtype`
            object, `numpy` char string, or a numpy scalar type.
            Default: None
        order (str, optional): memory layout for the array "C" or "F".
            Default: "C"
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
        sycl_queue (:class:`dpctl.SyclQueue`, optional): Not supported.

    Raises:
        errors.TypingError: If couldn't parse input types to dpnp.ones_like().
        errors.TypingError: If shape is provided.

    Returns:
        function: Local function `impl_dpnp_ones_like()`
    """

    if shape:
        raise errors.TypingError(
            "The parameter shape is not supported "
            + "inside overloaded dpnp.ones_like() function."
        )
    _ndim = x.ndim if hasattr(x, "ndim") and x.ndim is not None else 0
    _dtype = _parse_dtype(dtype, data=x)
    _order = x.layout if order is None else order
    _usm_type = _parse_usm_type(usm_type) if usm_type is not None else "device"
    _device = (
        _parse_device_filter_string(device) if device is not None else "unknown"
    )
    ret_ty = build_dpnp_ndarray(
        _ndim,
        layout=_order,
        dtype=_dtype,
        usm_type=_usm_type,
        device=_device,
        queue=sycl_queue,
    )
    if ret_ty:

        def impl(
            x,
            dtype=None,
            order="C",
            device=None,
            usm_type=None,
            sycl_queue=None,
        ):
            return impl_dpnp_ones_like(
                x,
                _dtype,
                _order,
                _device,
                _usm_type,
                sycl_queue,
                ret_ty,
            )

        return impl
    else:
        raise errors.TypingError(
            "Cannot parse input types to "
            + f"function dpnp.empty_like({x}, {dtype}, ...)."
        )
