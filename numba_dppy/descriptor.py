from __future__ import print_function, division, absolute_import
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions

from numba.core import dispatcher, utils, typing
from .target import DPPYTargetContext, DPPYTypingContext

from numba.core.cpu import CPUTargetOptions


class DPPYTarget(TargetDescriptor):
    options = CPUTargetOptions
    #typingctx = DPPYTypingContext()
    #targetctx = DPPYTargetContext(typingctx)

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPYTargetContext(self.typing_context)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return DPPYTypingContext()

    @property
    def target_context(self):
        """
        The target context for DPPY targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for DPPY targets.
        """
        return self._toplevel_typing_context



# The global DPPY target
dppy_target = DPPYTarget()
