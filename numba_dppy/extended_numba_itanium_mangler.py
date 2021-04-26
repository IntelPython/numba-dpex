from numba.core import itanium_mangler, types
from numba_dppy import target

EXQ2ABI = {
    target.SPIR_PRIVATE_ADDRSPACE : "AS0",
    target.SPIR_GLOBAL_ADDRSPACE : "AS1",
    target.SPIR_CONSTANT_ADDRSPACE : "AS2",
    target.SPIR_LOCAL_ADDRSPACE : "AS3",
    target.SPIR_GENERIC_ADDRSPACE : "AS4",
}


def mangle_type_or_value(typ):
    """
    Mangle type parameter and arbitrary value.
    This function extends Numba provided `magle_type_or_value()` to
    support numba.types.CPointer type.
    e.g. `int *` -> "Pi"
    For extended qualifier like `addrspace`, we generate "U" representing
    the presence of an extended qualifier. The actual address space is treated
    the same way as an identifier.
    e.g. `int (address_space(1)) *` -> "PU3AS1i"
    """
    if isinstance(typ, types.CPointer):
        rc = "P"
        if typ.addrspace is not None:
            rc += "U" + itanium_mangler.mangle_identifier(EXQ2ABI[typ.addrspace])
        rc += itanium_mangler.mangle_type_or_value(typ.dtype)
        return rc
    else:
        return itanium_mangler.mangle_type_or_value(typ)

mangle_type = mangle_type_or_value

def mangle_args(argtys):
    """
    Mangle sequence of Numba type objects and arbitrary values.
    """
    return ''.join([mangle_type_or_value(t) for t in argtys])



def mangle(ident, argtys):
    """
    Mangle identifier with Numba type objects and arbitrary values.
    """
    return itanium_mangler.PREFIX +  itanium_mangler.mangle_identifier(ident) + mangle_args(argtys)


