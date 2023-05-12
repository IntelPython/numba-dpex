# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import compiler, dispatcher, sigutils, utils
from numba.core.target_extension import dispatcher_registry, target_registry

from numba_dpex import config
from numba_dpex.core.caching import LRUCache, NullCache
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME
from numba_dpex.core.utils import (
    build_key,
    create_func_hash,
    strip_usm_metadata,
)

from .descriptor import dpex_target


class DpjitDispatcher(dispatcher.Dispatcher):
    """A dpex.djit-specific dispatcher.

    The DpjitDispatcher sets the targetdescr string to "dpex" so that Numba's
    Dispatcher can lookup the global target_registry with that string and
    correctly use the DpexTarget context.

    In addition, the dispatcher uses the `target_override` feature to set the
    target to dpex for every use of dpjit.

    """

    targetdescr = dpex_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        impl_kind="direct",
        pipeline_class=compiler.Compiler,
    ):
        from numba.core.target_extension import target_override

        with target_override("dpex"):
            dispatcher.Dispatcher.__init__(
                self,
                py_func,
                locals=locals,
                targetoptions=targetoptions,
                impl_kind=impl_kind,
                pipeline_class=pipeline_class,
            )
            self.py_func = py_func

    def enable_caching(self):
        self._cache_enabled = True
        if not config.ENABLE_CACHE:
            self._dpjit_cache = NullCache()
        elif self._cache_enabled:
            self._dpjit_cache = LRUCache(
                name="DpjitTargetCache",
                capacity=config.CACHE_SIZE,
                pyfunc=self.py_func,
            )
            self._func_hash = create_func_hash(self.py_func)
        else:
            self._dpjit_cache = NullCache()

    def compile(self, sig):
        if self._cache_enabled:
            argtypes, _ = sigutils.normalize_signature(sig)
            stripped_argtypes = strip_usm_metadata(argtypes)
            codegen_magic_tuple = self.targetctx.codegen().magic_tuple()
            key = build_key(
                stripped_argtypes, codegen_magic_tuple, self._func_hash
            )
            cres = self._dpjit_cache.get(key)
            if cres is None:
                self._cache_misses[sig] += 1
                cres = super().compile(sig)
                self._dpjit_cache.put(key, cres)
            else:
                self._cache_hits[sig] += 1
        else:
            cres = super().compile(sig)
        return cres


dispatcher_registry[target_registry[DPEX_TARGET_NAME]] = DpjitDispatcher
