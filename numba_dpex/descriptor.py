# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import utils
from numba.core.cpu import CPUTargetOptions
from numba.core.descriptors import TargetDescriptor

from .target import DPEX_TARGET_NAME, DpexTargetContext, DpexTypingContext


class DpexTarget(TargetDescriptor):
    options = CPUTargetOptions

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DpexTargetContext(self.typing_context, self._target_name)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DpexTypingContext()

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


# The global Dpex target
dpex_target = DpexTarget(DPEX_TARGET_NAME)
