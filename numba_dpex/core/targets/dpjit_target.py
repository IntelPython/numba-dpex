# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Defines the target and typing contexts for numba_dpex's dpjit decorator.
"""

from functools import cached_property

from numba.core import utils
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.core.cpu import CPUContext
from numba.core.imputils import Registry, RegistryLoader
from numba.core.target_extension import CPU, target_registry


class Dpex(CPU):
    pass


DPEX_TARGET_NAME = "dpex"

# Register a target hierarchy token in Numba's target registry, this
# permits lookup and reference in user space by the string "dpex"
target_registry[DPEX_TARGET_NAME] = Dpex

# This is the function registry for the dpu, it just has one registry, this one!
dpex_function_registry = Registry()


class DpexTargetContext(CPUContext):
    def __init__(self, typingctx, target=DPEX_TARGET_NAME):
        super().__init__(typingctx, target)

    @global_compiler_lock
    def init(self):
        self.is32bit = utils.MACHINE_BITS == 32
        self._internal_codegen = JITCPUCodegen("numba.exec")
        self.lower_extensions = {}
        # TODO: initialize nrt once switched to nrt from drt. Most likely we
        # call it somewhere. Double check.
        # https://github.com/IntelPython/numba-dpex/issues/1175
        # Initialize NRT runtime
        # rtsys.initialize(self) # noqa: E800
        self.refresh()

    @cached_property
    def dpexrt(self):
        from numba_dpex.core.runtime.context import DpexRTContext

        return DpexRTContext(self)

    def refresh(self):
        registry = dpex_function_registry
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.install_registry(registry)
        # Also refresh typing context, since @overload declarations can
        # affect it.
        self.typing_context.refresh()
        super().refresh()
