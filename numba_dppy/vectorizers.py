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

from numba.np.ufunc import deviceufunc
import numba_dppy as dppy
from numba_dppy.dppy_offload_dispatcher import DppyOffloadDispatcher

vectorizer_stager_source = """
def __vectorized_{name}({args}, __out__):
    __tid__ = __dppy__.get_global_id(0)
    if __tid__ < __out__.shape[0]:
        __out__[__tid__] = __core__({argitems})
"""


class DPPYVectorize(deviceufunc.DeviceVectorize):
    def _compile_core(self, sig):
        devfn = dppy.func(sig)(self.pyfunc)
        return devfn, devfn.cres.signature.return_type

    def _get_globals(self, corefn):
        glbl = self.pyfunc.__globals__.copy()
        glbl.update({"__dppy__": dppy, "__core__": corefn})
        return glbl

    def _compile_kernel(self, fnobj, sig):
        return dppy.kernel(fnobj)

    def build_ufunc(self):
        return DppyOffloadDispatcher(self.pyfunc)

    @property
    def _kernel_template(self):
        return vectorizer_stager_source
