# Copyright 2022 Intel Corporation
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

import hashlib
import warnings

from numba.core.caching import IndexDataCacheFile, _Cache, _CacheImpl

# from numba.core.errors import NumbaWarning
from numba.core.serialize import dumps

from numba_dpex.core import compiler

# from numba_dpex.core.kernel_interface.spirv_kernel import SpirvKernel


class SpirvKernelCacheImpl(_CacheImpl):
    def reduce(self, data):
        # TODO: Implement
        # return cres._reduce()
        # return (str, str, data)
        # return kernel._reduce_states()
        pass

    def rebuild(self, target_context, payload):
        # TODO: Implement _rebuild()
        # return compiler.CompileResult._rebuild(target_context, *payload)
        # return SpirvKernel._rebuild(**payload)
        pass

    def check_cachable(self, cres):
        # For the time being, assuming all numba-dpex Kernels are always cachable.
        # cannot_cache = None
        # if any(not x.can_cache for x in cres.lifted):
        #     cannot_cache = "as it uses lifted code"
        # elif cres.library.has_dynamic_globals:
        #     cannot_cache = (
        #         "as it uses dynamic globals "
        #         "(such as ctypes pointers and large global arrays)"
        #     )
        # if cannot_cache:
        #     msg = 'Cannot cache compiled function "%s" %s' % (
        #         cres.fndesc.qualname.split(".")[-1],
        #         cannot_cache,
        #     )
        #     warnings.warn_explicit(
        #         msg, NumbaWarning, self._locator._py_file, self._lineno
        #     )
        #     return False
        return True


class SpirvKernelCache(_Cache):
    """
    Implements a cache that saves and loads CUDA kernels and compile results.
    """

    _impl_class = SpirvKernelCacheImpl

    def __init__(self, py_func):
        self._name = repr(py_func)
        self._py_func = py_func
        self._impl = self._impl_class(py_func)
        self._cache_path = self._impl.locator.get_cache_path()
        # This may be a bit strict but avoids us maintaining a magic number
        source_stamp = self._impl.locator.get_source_stamp()
        filename_base = self._impl.filename_base
        self._cache_file = IndexDataCacheFile(
            cache_path=self._cache_path,
            filename_base=filename_base,
            source_stamp=source_stamp,
        )
        self.enable()

    @property
    def cache_path(self):
        return self._cache_path

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def flush(self):
        self._cache_file.flush()

    def load_overload(self, sig, target_context):
        print("numba_dpex.caching.SpirvKernelCache.load_overload().1")
        if not self._enabled:
            print("numba_dpex.caching.SpirvKernelCache.load_overload().2")
            return
        key = self._index_key(sig, target_context.codegen())
        data = self._cache_file.load(key)
        print("numba_dpex.caching.SpirvKernelCache.load_overload().3")
        return data

    def save_overload(self, sig, data, target_context):
        if not self._enabled:
            return
        if not self._impl.check_cachable(data):
            return
        self._impl.locator.ensure_cache_path()
        key = self._index_key(sig, target_context.codegen())
        # data = self._impl.reduce(data)
        self._cache_file.save(key, data)

    def _index_key(self, sig, codegen):
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """
        codebytes = self._py_func.__code__.co_code
        if self._py_func.__closure__ is not None:
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            # Note: cloudpickle serializes a function differently depending
            #       on how the process is launched; e.g. multiprocessing.Process
            cvarbytes = dumps(cvars)
        else:
            cvarbytes = b""

        return (
            sig,
            codegen.magic_tuple(),
            # device
            # backend
            (
                hashlib.sha256(codebytes).hexdigest(),
                hashlib.sha256(cvarbytes).hexdigest(),
            ),
        )
