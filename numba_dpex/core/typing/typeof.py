# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dpctl.tensor import usm_ndarray
from numba.extending import typeof_impl
from numba.np import numpy_support

from numba_dpex.core.types.usm_ndarray_type import USMNdArray
from numba_dpex.utils import address_space


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

    try:
        device = val.sycl_device.filter_string
    except AttributeError:
        raise ValueError("The device for the usm_ndarray could not be inferred")

    return USMNdArray(
        dtype=dtype,
        ndim=val.ndim,
        layout=layout,
        readonly=readonly,
        usm_type=usm_type,
        device=device,
        addrspace=address_space.GLOBAL,
    )
