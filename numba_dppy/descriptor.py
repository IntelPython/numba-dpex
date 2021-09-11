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

from numba.core import dispatcher, typing, utils
from numba.core.cpu import CPUTargetOptions
from numba.core.descriptors import TargetDescriptor
from numba.core.options import TargetOptions

from .target import DPPY_TARGET_NAME, DPPYTargetContext, DPPYTypingContext


class DPPYTarget(TargetDescriptor):
    options = CPUTargetOptions

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPPYTargetContext(self.typing_context, self._target_name)

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
dppy_target = DPPYTarget(DPPY_TARGET_NAME)
