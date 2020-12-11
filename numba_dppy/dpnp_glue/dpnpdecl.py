from numba.core.typing.templates import (AttributeTemplate, infer_getattr)
import numba_dppy
from numba import types
from numba.core.types.misc import RawPointer

@infer_getattr
class DppyDpnpTemplate(AttributeTemplate):
    key = types.Module(numba_dppy)

    def resolve_dpnp(self, mod):
        return types.Module(numba_dppy.dpnp)

@infer_getattr
class ArrayAttribute(AttributeTemplate):
    key = types.Array

    def resolve_shapeptr(self, ary):
        return types.voidptr
