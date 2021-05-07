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

from numba.core import itanium_mangler, types
from numba_dppy import target


def mangle_type_or_value(typ):
    """
    Mangle type parameter and arbitrary value.

    This function extends Numba's `magle_type_or_value()` to
    support numba.types.CPointer type, e.g. an ``int *`` argument will be
    mangled to "Pi".
    Mangling of extended qualifiers is supported only
    for address space qualifiers. In which case, the mangling
    follows the rule defined in Section 5.1.5.1 of the ``Itanium ABI
    <https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangle.qualified-type>``_.
    For example, an ``int global *`` argument will be mangeled to "PU3AS1i".

    Args:
        typ (numba.types, int, str) : Type to mangle

    Returns:
        str: The mangled name of the type

    """
    if isinstance(typ, types.CPointer):
        rc = "P"
        if typ.addrspace is not None:
            rc += "U" + itanium_mangler.mangle_identifier("AS" + str(typ.addrspace))
        rc += itanium_mangler.mangle_type_or_value(typ.dtype)
        return rc
    else:
        return itanium_mangler.mangle_type_or_value(typ)


mangle_type = mangle_type_or_value


def mangle_args(argtys):
    """
    Mangle sequence of Numba type objects and arbitrary values.
    """
    return "".join([mangle_type_or_value(t) for t in argtys])


def mangle(ident, argtys):
    """
    Mangle identifier with Numba type objects and arbitrary values.
    """
    return (
        itanium_mangler.PREFIX
        + itanium_mangler.mangle_identifier(ident)
        + mangle_args(argtys)
    )
