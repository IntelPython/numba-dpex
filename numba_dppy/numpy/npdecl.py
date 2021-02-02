from numba.core.typing.templates import AttributeTemplate, infer_getattr
import numba_dppy
from numba import types


@infer_getattr
class DppyNumpyTemplate(AttributeTemplate):
    key = types.Module(numba_dppy)

    def resolve_numpy(self, mod):
        return types.Module(numba_dppy.numpy)
