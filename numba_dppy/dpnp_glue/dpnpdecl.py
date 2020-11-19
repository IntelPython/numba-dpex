from numba.core.typing.templates import (AttributeTemplate, infer, infer_getattr, AbstractTemplate, signature)
import numba
from numba import types


@infer
class DpnpSum(AbstractTemplate):
    key = numba.dppl.dpnp.sum

    def generic(self, args, kws):
        assert not kws
        return_type =  args[0].dtype
        return signature(return_type, *args)


@infer_getattr
class DpnpTemplate(AttributeTemplate):
    key = types.Module(numba.dppl.dpnp)

    def resolve_sum(self, mod):
        return types.Function(DpnpSum)

@infer_getattr
class DpplDpnpTemplate(AttributeTemplate):
    key = types.Module(numba.dppl)

    def resolve_dpnp(self, mod):
        return types.Module(numba.dppl.dpnp)
