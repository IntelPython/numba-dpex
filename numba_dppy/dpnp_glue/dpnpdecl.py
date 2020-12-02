from numba.core.typing.templates import (AttributeTemplate, infer, infer_getattr, AbstractTemplate, signature)
import numba_dppy
from numba import types

@infer_getattr
class DpplDpnpTemplate(AttributeTemplate):
    key = types.Module(numba_dppy)

    def resolve_dpnp(self, mod):
        return types.Module(numba_dppy.dpnp)
