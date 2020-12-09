from numba.core.typing.templates import (AttributeTemplate, infer_getattr)
import numba_dppy
from numba import types

@infer_getattr
class DppyDpnpTemplate(AttributeTemplate):
    key = types.Module(numba_dppy)

    def resolve_dpnp(self, mod):
        return types.Module(numba_dppy.dpnp)
