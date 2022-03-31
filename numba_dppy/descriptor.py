# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
