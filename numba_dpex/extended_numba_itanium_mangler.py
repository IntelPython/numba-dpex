# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import itanium_mangler, types


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
            rc += "U" + itanium_mangler.mangle_identifier(
                "AS" + str(typ.addrspace)
            )
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


def mangle(ident, argtys, *, abi_tags=()):
    """
    Mangle identifier with Numba type objects and abi-tags.
    """
    kwargs = {}

    # for support numba 0.54 and <=0.55.0dev0=*_469
    if abi_tags:
        kwargs["abi_tags"] = abi_tags

    return (
        itanium_mangler.PREFIX
        + itanium_mangler.mangle_identifier(ident, **kwargs)
        + mangle_args(argtys)
    )
