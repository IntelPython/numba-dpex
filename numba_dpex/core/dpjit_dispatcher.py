# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import dispatcher, errors
from numba.core.target_extension import (
    dispatcher_registry,
    target_override,
    target_registry,
)

from numba_dpex.core.pipelines import dpjit_compiler
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME

from .descriptor import dpex_target


class _DpjitCompiler(dispatcher._FunctionCompiler):
    """A special compiler class used to compile numba_dpex.dpjit decorated
    functions.
    """

    def _compile_cached(self, args, return_type):
        # follows the same logic as original one, but triggers _compile_core
        # with dpex target overload.
        key = tuple(args), return_type
        try:
            return False, self._failed_cache[key]
        except KeyError:
            pass

        try:
            with target_override(DPEX_TARGET_NAME):
                retval = self._compile_core(args, return_type)
        except errors.TypingError as e:
            self._failed_cache[key] = e
            return False, e
        else:
            return True, retval


class DpjitDispatcher(dispatcher.Dispatcher):
    """A dpex.djit-specific dispatcher.

    The DpjitDispatcher sets the targetdescr string to "dpex" so that Numba's
    Dispatcher can lookup the global target_registry with that string and
    correctly use the DpexTarget context.

    """

    targetdescr = dpex_target

    def __init__(
        self,
        py_func,
        locals={},
        targetoptions={},
        pipeline_class=dpjit_compiler.DpjitCompiler,
    ):
        super().__init__(
            py_func=py_func,
            locals=locals,
            targetoptions=targetoptions,
            pipeline_class=pipeline_class,
        )
        self._compiler = _DpjitCompiler(
            py_func,
            self.targetdescr,
            targetoptions,
            locals,
            pipeline_class,
        )


dispatcher_registry[target_registry[DPEX_TARGET_NAME]] = DpjitDispatcher
