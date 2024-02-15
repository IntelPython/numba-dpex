# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""A new target descriptor that includes experimental features that should
eventually move into the numba_dpex.core.
"""

from functools import cached_property

from numba.core.descriptors import TargetDescriptor
from numba.core.target_extension import GPU, target_registry

from numba_dpex._kernel_api_impl.spirv.target import (
    SPIRVTargetContext,
    SPIRVTypingContext,
)
from numba_dpex.core.descriptor import DpexTargetOptions
from numba_dpex.experimental.models import exp_dmm


#  pylint: disable=R0903
class SyclDeviceExp(GPU):
    """Mark the hardware target as SYCL Device."""


DPEX_KERNEL_EXP_TARGET_NAME = "dpex_kernel_exp"

target_registry[DPEX_KERNEL_EXP_TARGET_NAME] = SyclDeviceExp


class DpexExpKernelTypingContext(SPIRVTypingContext):
    """Experimental typing context class extending the DpexKernelTypingContext
    by overriding super class functions for new experimental types.

    A new experimental type may require updating type inference for that type
    when it is used as an argument, value or attribute in a JIT compiled
    function. All such experimental functionality should be added here till they
    are stable enough to be migrated to DpexKernelTypingContext.
    """


#  pylint: disable=W0223
# FIXME: Remove the pylint disablement once we add an override for
# get_executable
class DpexExpKernelTargetContext(SPIRVTargetContext):
    """Experimental target context class extending the DpexKernelTargetContext
    by overriding super class functions for new experimental types.

    A new experimental type may require specific ways for handling the lowering
    to LLVM IR. All such experimental functionality should be added here till
    they are stable enough to be migrated to DpexKernelTargetContext.
    """

    allow_dynamic_globals = True

    def __init__(self, typingctx, target=DPEX_KERNEL_EXP_TARGET_NAME):
        super().__init__(typingctx, target)
        self.data_model_manager = exp_dmm


class DpexExpKernelTarget(TargetDescriptor):
    """
    Implements a target descriptor for numba_dpex.kernel decorated functions.
    """

    options = DpexTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        """Lazily-initialized top-level target context, for all threads."""
        return DpexExpKernelTargetContext(
            self.typing_context, self._target_name
        )

    @cached_property
    def _toplevel_typing_context(self):
        """Lazily-initialized top-level typing context, for all threads."""
        return DpexExpKernelTypingContext()

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


# A global instance of the DpexKernelTarget with the experimental features
dpex_exp_kernel_target = DpexExpKernelTarget(DPEX_KERNEL_EXP_TARGET_NAME)
