# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Provides helper functions to convert from numba types to numba_dpex types.

Currently the module supports the following converter functions:
    - npytypes_array_to_dppy_array: types.npytypes.Array to
                                    numba_dpex.dppy_array_type.DPPYArray.

"""
from numba.core import types
from numba.np import numpy_support
from numba_dpex import dppy_array_type

from .array_utils import get_info_from_suai
from .constants import address_space

__all__ = ["npytypes_array_to_dppy_array", "suai_to_dppy_array_type"]


def npytypes_array_to_dppy_array(arrtype, addrspace=address_space.GLOBAL):
    """Convert   Numba's Array type to numba_dpex's DPPYArray type.

    Numba's ``Array`` type does not have a notion of address space for the data
    pointer. numba_dpex defines its own array type, DPPYArray, that is similar
    to Numba's Array, but the data pointer has an associated address space.
    In addition, the ``meminfo`` and the ``parent`` attributes of ``Array``
    are stored as ``CPointer`` types instead of ``PyObject``. The converter
    function converts the Numba ``Array`` type to ``DPPYArray`` type with
    address space of pointer members typed to the specified address space.

    Args:
        arrtype (numba.types): A numba data type that should be
            ``numba.types.Array``.
        specified: Defaults to ``numba_dpex.utils.address_space.GLOBAL``.
            The SPIR-V address space to which the data pointer of the array
            belongs.

    Returns: The numba_dpex data type corresponding to the input numba type.

    Raises:
        NotImplementedError: If the input numba type is not
                             ``numba.types.Array``

    """
    # We are not using isinstance() here as we want to
    # strictly match with types.Array. There are numba-dpex types
    # that inherit from types.Array and those type would
    # also get trapped if isinstance() is used.
    if type(arrtype) is types.npytypes.Array:
        return dppy_array_type.DPPYArray(
            arrtype.dtype,
            arrtype.ndim,
            arrtype.layout,
            not arrtype.mutable,
            arrtype.name,
            arrtype.aligned,
            addrspace=addrspace,
        )
    else:
        raise NotImplementedError


def suai_to_dppy_array_type(arr, addrspace=address_space.GLOBAL):
    """Create type for Array with __sycl_usm_array_interface__ (SUAI) attribute.

    This function cretes a Numba type for arrays with SUAI attribute.

    Args:
        arr: Array with SUAI attribute.
        addrspace: Address space this array is allocated in.

    Returns: The Numba type for SUAI array.

    Raises:
        NotImplementedError: If the dtype of the passed array is not supported.
    """
    from numba_dpex.dpctl_iface import USMNdArrayType

    (
        usm_mem,
        total_size,
        shape,
        ndim,
        itemsize,
        strides,
        dtype,
    ) = get_info_from_suai(arr)

    try:
        dtype = numpy_support.from_dtype(dtype)
    except NotImplementedError:
        raise ValueError("Unsupported array dtype: %s" % (dtype,))

    layout = "C"
    readonly = False

    return USMNdArrayType(
        dtype,
        ndim,
        layout,
        None,
        readonly,
        addrspace=addrspace,
    )
