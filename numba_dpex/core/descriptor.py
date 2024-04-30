# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from numba.core import options, targetconfig, typing
from numba.core.cpu import CPUTargetOptions
from numba.core.descriptors import TargetDescriptor

from numba_dpex.core import config
from numba_dpex.kernel_api_impl.spirv.target import (
    SPIRV_TARGET_NAME,
    CompilationMode,
    SPIRVTargetContext,
    SPIRVTypingContext,
)

from .targets.dpjit_target import (
    DPEX_TARGET_NAME,
    DpexTargetContext,
    DpexTypingContext,
)

_option_mapping = options._mapping


def _inherit_if_not_set(flags, options, name, default=targetconfig._NotSet):
    if name in options:
        setattr(flags, name, options[name])
        return

    cstk = targetconfig.ConfigStack()
    if cstk:
        # inherit
        top = cstk.top()
        if hasattr(top, name):
            setattr(flags, name, getattr(top, name))
            return

    if default is not targetconfig._NotSet:
        setattr(flags, name, default)


class DpexTargetOptions(CPUTargetOptions):
    experimental = _option_mapping("experimental")
    release_gil = _option_mapping("release_gil")
    no_compile = _option_mapping("no_compile")
    inline_threshold = _option_mapping("inline_threshold")
    _compilation_mode = _option_mapping("_compilation_mode")
    # TODO: create separate parfor kernel target
    _parfor_body_args = _option_mapping("_parfor_body_args")

    def finalize(self, flags, options):
        super().finalize(flags, options)
        _inherit_if_not_set(flags, options, "experimental", False)
        _inherit_if_not_set(flags, options, "release_gil", False)
        _inherit_if_not_set(flags, options, "no_compile", True)
        if config.INLINE_THRESHOLD is not None:
            _inherit_if_not_set(
                flags, options, "inline_threshold", config.INLINE_THRESHOLD
            )
        else:
            _inherit_if_not_set(flags, options, "inline_threshold", 0)
        _inherit_if_not_set(
            flags, options, "_compilation_mode", CompilationMode.KERNEL
        )
        _inherit_if_not_set(flags, options, "_parfor_body_args", None)


class DpexKernelTarget(TargetDescriptor):
    """
    Implements a target descriptor for numba_dpex.kernel decorated functions.
    """

    options = DpexTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        """Lazily-initialized top-level target context, for all threads."""
        return SPIRVTargetContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        """Lazily-initialized top-level typing context, for all threads."""
        return SPIRVTypingContext()

    @property
    def target_context(self):
        """
        The target context used by the Dpex compiler pipeline.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for used by the Dpex compiler pipeline.
        """
        return self._toplevel_typing_context

    @property
    def target_name(self):
        return self._target_name


class DpexTarget(TargetDescriptor):
    """
    Implements a target descriptor for numba_dpex.dpjit decorated functions.
    """

    options = CPUTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DpexTargetContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DpexTypingContext()

    @property
    def target_context(self):
        """
        The target context for dpex targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for dpex targets.
        """
        return self._toplevel_typing_context


# A global instance of the DpexKernelTarget
dpex_kernel_target = DpexKernelTarget(SPIRV_TARGET_NAME)

# A global instance of the DpexTarget
dpex_target = DpexTarget(DPEX_TARGET_NAME)
