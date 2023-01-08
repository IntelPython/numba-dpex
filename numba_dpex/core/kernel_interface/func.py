# SPDX-FileCopyrightText: 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""_summary_
"""


from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate

from numba_dpex.core.compiler import compile_with_dpex
from numba_dpex.core.descriptor import dpex_target


def compile_func(pyfunc, return_type, args, debug=None):
    cres = compile_with_dpex(
        pyfunc=pyfunc,
        pyfunc_name=pyfunc.__name__,
        return_type=return_type,
        target_context=dpex_target.target_context,
        typing_context=dpex_target.typing_context,
        args=args,
        is_kernel=False,
        debug=debug,
    )
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    cres.target_context.mark_ocl_device(func)
    devfn = DpexFunction(cres)

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


class DpexFunctionTemplate(object):
    """Unmaterialized dpex function"""

    def __init__(self, pyfunc, debug=None):
        self.py_func = pyfunc
        self.debug = debug
        self._compileinfos = {}

    def compile(self, args):
        """Compile a dpex.func decorated Python function with the given
        argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.
        """
        if args not in self._compileinfos:
            cres = compile_with_dpex(
                pyfunc=self.py_func,
                pyfunc_name=self.py_func.__name__,
                return_type=None,
                target_context=dpex_target.target_context,
                typing_context=dpex_target.typing_context,
                args=args,
                is_kernel=False,
                debug=self.debug,
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

        else:
            cres = self._compileinfos[args]

        return cres.signature


class DpexFunction(object):
    def __init__(self, cres):
        self.cres = cres
