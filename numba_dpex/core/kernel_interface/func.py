# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""_summary_
"""


from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate

from numba_dpex import config
from numba_dpex.core.caching import LRUCache, NullCache, build_key
from numba_dpex.core.compiler import compile_with_dpex
from numba_dpex.core.descriptor import dpex_target


class DpexFunction(object):
    def __init__(self, pyfunc, return_type, debug=None):
        self._pyfunc = pyfunc
        self._return_type = return_type
        self._debug = debug
        self._enable_cache = True

        if not config.ENABLE_CACHE:
            self._cache = NullCache()
        elif self._enable_cache:
            self._cache = LRUCache(
                capacity=config.CACHE_SIZE, pyfunc=self._pyfunc
            )
        else:
            self._cache = NullCache()
        self._cache_hits = 0

    @property
    def cache(self):
        return self._cache

    @property
    def cache_hits(self):
        return self._cache_hits

    def compile(self, args):
        argtypes = [
            dpex_target.typing_context.resolve_argument_type(arg)
            for arg in args
        ]
        key = build_key(
            tuple(argtypes),
            self._pyfunc,
            dpex_target.target_context.codegen(),
        )
        breakpoint()
        cres = self._cache.get(key)
        if cres is None:
            self._cache_hits += 1
            print("----------> DpexFunction hit")
            cres = compile_with_dpex(
                pyfunc=self._pyfunc,
                pyfunc_name=self._pyfunc.__name__,
                return_type=self._return_type,
                target_context=dpex_target.target_context,
                typing_context=dpex_target.typing_context,
                args=args,
                is_kernel=False,
                debug=self._debug,
            )
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_ocl_device(func)
            self._cache.put(key, cres)

        return cres


class DpexFunctionTemplate(object):
    """Unmaterialized dpex function"""

    def __init__(self, pyfunc, debug=None, enable_cache=True):
        self._pyfunc = pyfunc
        self._debug = debug
        self._enable_cache = enable_cache
        self._compileinfos = {}

        if not config.ENABLE_CACHE:
            self._cache = NullCache()
        elif self._enable_cache:
            self._cache = LRUCache(
                capacity=config.CACHE_SIZE, pyfunc=self._pyfunc
            )
        else:
            self._cache = NullCache()
        self._cache_hits = 0

    @property
    def cache(self):
        return self._cache

    @property
    def cache_hits(self):
        return self._cache_hits

    def compile(self, args):
        """Compile a dpex.func decorated Python function with the given
        argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.
        """
        argtypes = [
            dpex_target.typing_context.resolve_argument_type(arg)
            for arg in args
        ]
        key = build_key(
            tuple(argtypes),
            self._pyfunc,
            dpex_target.target_context.codegen(),
        )
        breakpoint()
        cres = self._cache.get(key)
        if cres is None:
            self._cache_hits += 1
            print("----------> DpexFunctionTemplate hit")
            cres = compile_with_dpex(
                pyfunc=self._pyfunc,
                pyfunc_name=self._pyfunc.__name__,
                return_type=None,
                target_context=dpex_target.target_context,
                typing_context=dpex_target.typing_context,
                args=args,
                is_kernel=False,
                debug=self._debug,
            )
            func = cres.library.get_function(cres.fndesc.llvm_func_name)
            cres.target_context.mark_ocl_device(func)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(
                    self, cres.fndesc, libs
                )
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)
            self._cache.put(key, cres)

        return cres.signature


def compile_func(pyfunc, return_type, args, debug=None):

    devfn = DpexFunction(pyfunc, return_type, debug=debug)
    cres = devfn.compile(args)

    class _function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, _function_template)
    libs = [cres.library]
    cres.target_context.insert_user_function(devfn, cres.fndesc, libs)
    return devfn


def compile_func_template(pyfunc, debug=None):
    """Compile a DpexFunctionTemplate"""

    dft = DpexFunctionTemplate(pyfunc, debug=debug)

    class _function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            if kws:
                raise AssertionError("No keyword arguments allowed.")
            return dft.compile(args)

    dpex_target.typing_context.insert_user_function(dft, _function_template)
    return dft
