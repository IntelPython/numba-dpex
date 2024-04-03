# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Defines the target and typing contexts for numba_dpex's dpjit decorator.
"""

from functools import cached_property

from numba.core import typing
from numba.core.compiler_lock import global_compiler_lock
from numba.core.cpu import CPUContext
from numba.core.imputils import Registry
from numba.core.target_extension import CPU, target_registry

from numba_dpex.core.datamodel.models import _init_dpjit_data_model_manager
from numba_dpex.dpctl_iface import dpctlimpl
from numba_dpex.dpnp_iface import dpnp_ufunc_db


class Dpex(CPU):
    pass


DPEX_TARGET_NAME = "dpex"

# Register a target hierarchy token in Numba's target registry, this
# permits lookup and reference in user space by the string "dpex"
target_registry[DPEX_TARGET_NAME] = Dpex

dpex_function_registry = Registry()


class DpexTypingContext(typing.Context):
    """Custom typing context to support dpjit compilation."""

    def load_additional_registries(self):
        """Register dpjit specific functions like dpnp ufuncs."""
        from numba_dpex.core.typing import dpnpdecl

        self.install_registry(dpnpdecl.registry)
        super().load_additional_registries()


class DpexTargetContext(CPUContext):
    def __init__(self, typingctx, target=DPEX_TARGET_NAME):
        super().__init__(typingctx, target)

    @global_compiler_lock
    def init(self):
        self.lower_extensions = {}
        super().init()

        self.data_model_manager = _init_dpjit_data_model_manager()

        # TODO: initialize nrt once switched to nrt from drt. Most likely we
        # call it somewhere. Double check.
        # https://github.com/IntelPython/numba-dpex/issues/1175
        # Initialize NRT runtime
        # rtsys.initialize(self) # noqa: E800

    @cached_property
    def dpexrt(self):
        from numba_dpex.core.runtime.context import DpexRTContext

        return DpexRTContext(self)

    def load_additional_registries(self):
        """
        Load dpjit-specific registries.
        """
        self.install_registry(dpex_function_registry)
        self.install_registry(dpctlimpl.registry)

        # loading CPU specific registries
        super().load_additional_registries()

    # TODO: do we need it?
    def get_ufunc_info(self, ufunc_key):
        return dpnp_ufunc_db.get_ufunc_info(ufunc_key)
