# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.np import numpy_support

from numba_dpex.core.types import USMNdArray
from numba_dpex.core.utils import get_info_from_suai
from numba_dpex.utils.constants import address_space


def to_usm_ndarray(suai_attrs, addrspace=address_space.GLOBAL):
    """Converts an array-like object that has the _sycl_usm_array_interface__
    attribute to numba_dpex.types.UsmNdArray.

    Args:
        suai_attrs: The extracted SUAI information for an array-like object.
        addrspace: Address space this array is allocated in.

    Returns: The Numba type for SUAI array.

    Raises:
        NotImplementedError: If the dtype of the passed array is not supported.
    """
    try:
        dtype = numpy_support.from_dtype(suai_attrs.dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (dtype,))

    # If converting from an unknown array-like object that implements
    # __sycl_usm_array_interface__, the layout is always hard-coded to
    # C-contiguous.
    layout = "C"

    return USMNdArray(
        dtype=dtype,
        ndim=suai_attrs.dimensions,
        layout=layout,
        usm_type=suai_attrs.usm_type,
        device=suai_attrs.device,
        queue=suai_attrs.queue,
        readonly=not suai_attrs.is_writable,
        name=None,
        aligned=True,
        addrspace=addrspace,
    )
