# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""_summary_
"""


from numba.core import sigutils, types
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate

from numba_dpex import config
from numba_dpex.core.caching import LRUCache, NullCache, build_key
from numba_dpex.core.compiler import compile_with_dpex
from numba_dpex.core.descriptor import dpex_target
from numba_dpex.utils import npytypes_array_to_dpex_array


class DpexFunction(object):
    """Class to materialize dpex function"""

    def __init__(self, pyfunc, debug=None):
        """Constructor for DpexFunction

        Args:
            pyfunc (function): A python function to be compiled.
            debug (object, optional): Debug option for compilation.
                Defaults to None.
        """
        self._pyfunc = pyfunc
        self._debug = debug

    def compile(self, arg_types, return_types):
        """The actual compilation function.

        Args:
            arg_types (tuple): Function argument types in a tuple.
            return_types (numba.core.types.scalars.Integer):
                An integer value to specify the return type.

        Returns:
            numba.core.compiler.CompileResult: The compiled result
        """

        cres = compile_with_dpex(
            pyfunc=self._pyfunc,
            pyfunc_name=self._pyfunc.__name__,
            return_type=return_types,
            target_context=dpex_target.target_context,
            typing_context=dpex_target.typing_context,
            args=arg_types,
            is_kernel=False,
            debug=self._debug,
        )
        func = cres.library.get_function(cres.fndesc.llvm_func_name)
        cres.target_context.mark_ocl_device(func)

        return cres


class DpexFunctionTemplate(object):
    """Unmaterialized dpex function"""

    def __init__(self, pyfunc, debug=None, enable_cache=True):
        """AI is creating summary for __init__

        Args:
            pyfunc (function): A python function to be compiled.
            debug (object, optional): Debug option for compilation.
                Defaults to None.
            enable_cache (bool, optional): Flag to turn on/off caching.
                Defaults to True.
        """
        self._pyfunc = pyfunc
        self._debug = debug
        self._enable_cache = enable_cache

        if not config.ENABLE_CACHE:
            self._cache = NullCache()
        elif self._enable_cache:
            self._cache = LRUCache(
                name="DpexFunctionTemplateCache",
                capacity=config.CACHE_SIZE,
                pyfunc=self._pyfunc,
            )
        else:
            self._cache = NullCache()
        self._cache_hits = 0

    @property
    def cache(self):
        """Cache accessor"""
        return self._cache

    @property
    def cache_hits(self):
        """Cache hit count accessor"""
        return self._cache_hits

    def compile(self, args):
        """Compile a dpex.func decorated Python function

        Compile a dpex.func decorated Python function with the given
        argument types. Each signature is compiled once by caching the
        compiled function inside this object.

        Args:
            args (tuple): Function argument types in a tuple.

        Returns:
            numba.core.typing.templates.Signature: Signature of the
                compiled result.
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
        cres = self._cache.get(key)
        if cres is None:
            self._cache_hits += 1
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
            libs = [cres.library]

            cres.target_context.insert_user_function(self, cres.fndesc, libs)
            # cres.target_context.add_user_function(self, cres.fndesc, libs)
            self._cache.put(key, cres)
        return cres.signature


def compile_func(pyfunc, signature, debug=None):
    """AI is creating summary for compile_func

    Args:
        pyfunc (function): A python function to be compiled.
        signature (list): A list of numba.core.typing.templates.Signature's
        debug (object, optional): Debug options. Defaults to None.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunction: DpexFunction object
    """

    devfn = DpexFunction(pyfunc, debug=debug)

    cres = []
    for sig in signature:
        arg_types, return_types = sigutils.normalize_signature(sig)
        arg_types = tuple(
            [
                npytypes_array_to_dpex_array(ty)
                if isinstance(ty, types.npytypes.Array)
                else ty
                for ty in arg_types
            ]
        )
        c = devfn.compile(arg_types, return_types)
        cres.append(c)

    class _function_template(ConcreteTemplate):
        unsafe_casting = False
        exact_match_required = True
        key = devfn
        cases = [c.signature for c in cres]

    # TODO: If context is only one copy, just make it global
    cres[0].typing_context.insert_user_function(devfn, _function_template)

    for c in cres:
        c.target_context.insert_user_function(devfn, c.fndesc, [c.library])

    return devfn


def compile_func_template(pyfunc, debug=None):
    """Compile a DpexFunctionTemplate

    Args:
        pyfunc (function): A python function to be compiled.
        debug (object, optional): Debug options. Defaults to None.

    Raises:
        AssertionError: Raised if keyword arguments are supplied in
            the inner generic function.

    Returns:
        numba_dpex.core.kernel_interface.func.DpexFunctionTemplate:
            A DpexFunctionTemplate object.
    """

    dft = DpexFunctionTemplate(pyfunc, debug=debug)

    class _function_template(AbstractTemplate):
        unsafe_casting = False
        exact_match_required = True
        key = dft

        # TODO: Talk with the numba team and see why this has been
        # called twice, could be a bug with numba.
        def generic(self, args, kws):
            if kws:
                raise AssertionError("No keyword arguments allowed.")
            return dft.compile(args)

    dpex_target.typing_context.insert_user_function(dft, _function_template)

    return dft
