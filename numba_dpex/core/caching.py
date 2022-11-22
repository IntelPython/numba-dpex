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


class SpirvKernelCacheImpl(_CacheImpl):
    """Implementation of `_CacheImpl` to be used by subclasses of `_Cache`.

    This class is an implementation of `_CacheImpl` to be used by subclasses of `_Cache`.
    To be assigned in `_impl_class`. Implements the more common and core mechanism for the
    caching.

    """

    def reduce(self, data):
        """Serialize an object before caching.

        Args:
            data (object): The object to be serialized before pickling.
        """
        # TODO: Implement, but looks like we might not need it at all.
        # Look at numba.core.caching for how to implement.
        pass

    def rebuild(self, target_context, reduced_data):
        """Deserialize after unpickling from the cache.

        Args:
            target_context (numba_dpex.core.target.DpexTargetContext):
                The target context for the kernel.
            reduced_data (object): The data to be deserialzed after unpickling.
        """
        # TODO: Implement, but looks like we might not need it at all.
        # Look at numba.core.caching for how to implement.
        pass

    def check_cachable(self, cres):
        """Check if a certain object is cacheable.

        Args:
            cres (object): The object to be cached. For example, if the object is
            `CompileResult`, then you might want to follow the similar checks as
            has been done in `numba.core.caching.CompileResultCacheImpl`.

        Returns:
            bool: Return `True` if cacheable, otherwise `False`.
        """
        # TODO: Although, for the time being, assuming all SPIR-V Kernels
        # are always cachable. However, we might need to add some bells and
        # whistles in the future. Look at numba.core.caching for how to implement.
        return True


class SpirvKernelCache(_Cache):
    """Implements a cache that saves and loads SPIR-V kernels and compile results.

    This class implements the ABC `_Cache`. Mainly constructs key-value pair
    for the data to be cached. The data has been saved/retrieved from the file
    using that key-value pair.
    """

    # _CacheImpl object to be used
    _impl_class = SpirvKernelCacheImpl

    def __init__(self, py_func):
        """Constructor for SprivKernelCache.

        Args:
            py_func (function): The python function of the corresponding
                                spirv kernel to be cached.
        """
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
        """Path to cache files.

        Returns:
            str: The path to cache files.
        """
        return self._cache_path

    def enable(self):
        """Enables caching."""
        self._enabled = True

    def disable(self):
        """Disables caching."""
        self._enabled = False

    def flush(self):
        """Flushes the buffer for cache file."""
        self._cache_file.flush()

    def load_overload(self, sig, target_context):
        """Loads the 'overload', i.e. kernel from cache.

        Args:
            sig (inspect.Signature): The signature object of a python function.
            target_context (numba_dpex.core.target.DpexTargetContext):
                The target context of the kernel.

        Returns:
            object: The unpickled object from the cache.
        """
        if not self._enabled:
            return
        key = self._index_key(sig, target_context.codegen())
        data = self._cache_file.load(key)
        return data

    def save_overload(self, sig, data, target_context):
        """Saves the 'overload', i.e. kernel into cache.

        Args:
            sig (inspect.Signature): The signature object of a python function.
            data (object): The object to be saved in the cache.
            target_context (numba_dpex.core.target.DpexTargetContext):
                The target context of the kernel.
        """
        if not self._enabled:
            return
        if not self._impl.check_cachable(data):
            return
        self._impl.locator.ensure_cache_path()
        key = self._index_key(sig, target_context.codegen())
        self._cache_file.save(key, data)

    def _index_key(self, sig, codegen):
        """Constructs a key from the data object.

        Compute index key for the given signature and codegen. It includes
        a description of the OS, target architecture and hashes of the bytecode
        for the function and, if the function has a __closure__, a hash of the
        cell_contents.

        Args:
            sig (inspect.Signature): The signature object of a python function.
            codegen (numba_dpex.codegen.JITSPIRVCodegen):
                The JITSPIRVCodegen found from the target context.

        Returns:
            tuple: A tuple of signature, magic_tuple of codegen and another tuple of
                    hashcodes from bytecode and cell_contents.
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
            # TODO: add device
            # TODO: add backend
            (
                hashlib.sha256(codebytes).hexdigest(),
                hashlib.sha256(cvarbytes).hexdigest(),
            ),
        )
