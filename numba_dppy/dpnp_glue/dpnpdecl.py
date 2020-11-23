from numba.core.typing.templates import (AttributeTemplate, infer, infer_getattr, AbstractTemplate, signature)
import numba
from numba import types

@infer_getattr
class DpplDpnpTemplate(AttributeTemplate):
    key = types.Module(numba.dppl)

    def resolve_dpnp(self, mod):
        return types.Module(numba.dppl.dpnp)
