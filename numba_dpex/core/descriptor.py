# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import utils
from numba.core.cpu import CPUTargetOptions
from numba.core.descriptors import TargetDescriptor

from .targets.kernel_target import (
    DPEX_KERNEL_TARGET_NAME,
    DpexKernelTargetContext,
    DpexKernelTypingContext,
)


class DpexKernelTarget(TargetDescriptor):
    options = CPUTargetOptions

    @utils.cached_property
    def _toplevel_target_context(self):
        """Lazily-initialized top-level target context, for all threads."""
        return DpexKernelTargetContext(self.typing_context, self._target_name)

    @utils.cached_property
    def _toplevel_typing_context(self):
        """Lazily-initialized top-level typing context, for all threads."""
        return DpexKernelTypingContext()

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


# A global instance of the DpexKernelTarget
dpex_kernel_target = DpexKernelTarget(DPEX_KERNEL_TARGET_NAME)
