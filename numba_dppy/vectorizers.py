from numba.np.ufunc import deviceufunc
import numba_dppy as dppy
from numba_dppy.dppy_offload_dispatcher import DppyOffloadDispatcher

vectorizer_stager_source = '''
def __vectorized_{name}({args}, __out__):
    __tid__ = __dppy__.get_global_id(0)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
'''


class DPPYVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        devfn = dppy.func(sig)(self.pyfunc)
        return devfn, devfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({'__dppy__': dppy,
                     '__core__': corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return dppy.kernel(fnobj)

    def build_ufunc(self):
        return DppyOffloadDispatcher(self.pyfunc)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source
