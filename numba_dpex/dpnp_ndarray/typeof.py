# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.extending import typeof_impl
from numba.np import numpy_support

from .types import dpnp_ndarray_Type, ndarray


# This tells Numba how to create a UsmSharedArrayType when a usmarray is passed
# into a njit function.
@typeof_impl.register(ndarray)
def typeof_dpnp_ndarray(val, c):
    try:
        dtype = numpy_support.from_dtype(val.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (val.dtype,))

    try:
        layout = numpy_support.map_layout(val)
    except AttributeError:
        try:
            # passing nested object as dpnp.ndarray does not support flags yet
            layout = numpy_support.map_layout(val._array_obj)
        except TypeError:
            layout = "C"

    try:
        readonly = not val.flags.writeable
    except AttributeError:
        # dpnp.ndarray does not support flags
        readonly = False

    return dpnp_ndarray_Type(dtype, val.ndim, layout, readonly=readonly)
