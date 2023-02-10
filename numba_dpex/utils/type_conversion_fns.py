# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

""" Provides helper functions to convert from numba types to dpex types.

Currently the module supports the following converter functions:
    - types.npytypes.Array to numba_dpex.core.types.Array.

"""
from numba.core import types

from numba_dpex.core.types import Array

from .constants import address_space

__all__ = ["npytypes_array_to_dpex_array"]


def npytypes_array_to_dpex_array(arrtype, addrspace=address_space.GLOBAL):
    """Converts Numba's ``Array`` type to ``numba_dpex.core.types.Array``
    type.

    Numba's ``Array`` type does not have a notion of address space for the data
    pointer. To get around the issues, numba_dpex defines its own array type
    that is inherits from Numba's Array type. In the
    ``numba_dpex.core.types.Array`` type the data pointer has an
    associated address space. In addition, the ``meminfo`` and the ``parent``
    attributes are stored as ``CPointer`` types instead of ``PyObject``.

    The converter function converts the Numba ``Array`` type to
    ``numba_dpex.core.types.Array`` type with address space of pointer
    members typed to the specified address space.

    Args:
        arrtype (numba.types): A numba data type that should be
            ``numba.types.Array``.
        specified: Defaults to ``numba_dpex.utils.address_space.GLOBAL``.
            The SPIR-V address space to which the data pointer of the array
            belongs.

    Returns: The dpex data type corresponding to the input numba type.

    Raises:
        NotImplementedError: If the input numba type is not
                             ``numba.types.Array``

    """
    # We are not using isinstance() here as we want to strictly match with
    # numba.types.Array and not with a subtype such as
    # numba_dpex.core.types.Array.
    if type(arrtype) is types.npytypes.Array:
        return Array(
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
