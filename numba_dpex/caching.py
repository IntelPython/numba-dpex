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

from numba.core.caching import IndexDataCacheFile, _Cache, _CacheImpl
from numba.core.serialize import dumps

from numba_dpex.core import compiler


class DpexCacheImpl(_CacheImpl):
    def reduce(self, cres):
        """
        Returns a serialized CompileResult
        """
        return cres._reduce()

    def rebuild(self, target_context, payload):
        """
        Returns the unserialized CompileResult
        """
        return compiler.CompileResult._rebuild(target_context, *payload)

    def check_cachable(self, cres):
        # For the time being, assuming all numba-dpex Kernels are always cachable.
        return True


class DpexCache(_Cache):
    """
    Implements a cache that saves and loads CUDA kernels and compile results.
    """

    _impl_class = DpexCacheImpl

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
        print("-----> caching.load_overload().here-1")
        if not self._enabled:
            print("-----> caching.load_overload().here-2")
            return
        key = self._index_key(sig, target_context.codegen())
        print("-----> caching.load_overload().here-3")
        # key = self._index_key(sig)
        data = self._cache_file.load(key)
        print("-----> caching.load_overload().here-4")
        if data is not None:
            print("-----> caching.load_overload().here-5")
            data = self._impl.rebuild(target_context, data)
        print("-----> caching.load_overload().here-6")
        return data

    def save_overload(self, sig, data):
        print("-----> caching.save_overload().here-1")
        if not self._enabled:
            print("-----> caching.save_overload().here-2")
            return
        if not self._impl.check_cachable(data):
            print("-----> caching.save_overload().here-3")
            return
        self._impl.locator.ensure_cache_path()
        print("-----> caching.save_overload().here-4")
        print(data.dump())
        key = self._index_key(sig, data.codegen)
        print("-----> caching.save_overload().here-5")
        # key = self._index_key(sig)
        data = self._impl.reduce(data)
        print("-----> caching.save_overload().here-6")
        self._cache_file.save(key, data)
        print("-----> caching.save_overload().here-7")

    def _index_key(self, sig, codegen):
        # def _index_key(self, sig):
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """
        print("-----> caching._index_key.codegen:", codegen)
        codebytes = self._py_func.__code__.co_code
        if self._py_func.__closure__ is not None:
            print("-----> caching._index_key.here-1")
            cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
            # Note: cloudpickle serializes a function differently depending
            #       on how the process is launched; e.g. multiprocessing.Process
            cvarbytes = dumps(cvars)
        else:
            print("-----> caching._index_key.here-2")
            cvarbytes = b""

        # hasher = lambda x: hashlib.sha256(x).hexdigest()
        # def hasher(x):
        #      return hashlib.sha256(x).hexdigest()
        return (
            sig,
            codegen.magic_tuple(),
            (
                # hasher(codebytes),
                hashlib.sha256(codebytes).hexdigest(),
                # hasher(cvarbytes),
                hashlib.sha256(cvarbytes).hexdigest(),
            ),
        )
