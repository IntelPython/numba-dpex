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

""" Provides helper functions to convert from numba types to numba_dppy types.

Currently the module supports the following converter functions:
    - npytypes_array_to_dppy_array: types.npytypes.Array to
                                    numba_dppy.dppy_array_type.DPPYArray.

"""
from numba.core import types
from numba_dppy import dppy_array_type
from .constants import address_space

__all__ = ["npytypes_array_to_dppy_array"]


def npytypes_array_to_dppy_array(arrtype, addrspace=address_space.GLOBAL):
    """Convert   Numba's Array type to numba_dppy's DPPYArray type.

    Numba's ``Array`` type does not have a notion of address space for the data
    pointer. numba_dppy defines its own array type, DPPYArray, that is similar
    to Numba's Array, but the data pointer has an associated address space.
    In addition, the ``meminfo`` and the ``parent`` attributes of ``Array``
    are stored as ``CPointer`` types instead of ``PyObject``. The converter
    function converts the Numba ``Array`` type to ``DPPYArray`` type with
    address space of pointer members typed to the specified address space.

    Args:
        arrtype (numba.types): A numba data type that should be
            ``numba.types.Array``.
        specified: Defaults to ``numba_dppy.utils.address_space.GLOBAL``.
            The SPIR-V address space to which the data pointer of the array
            belongs.

    Returns: The numba_dppy data type corresponding to the input numba type.

    Raises:
        NotImplementedError: If the input numba type is not
                             ``numba.types.Array``

    """
    if isinstance(arrtype, types.npytypes.Array):
        return dppy_array_type.DPPYArray(
            arrtype.dtype,
            arrtype.ndim,
            arrtype.layout,
            arrtype.py_type,
            not arrtype.mutable,
            arrtype.name,
            arrtype.aligned,
            addrspace=addrspace,
        )
    else:
        raise NotImplementedError
