# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Defines the target and typing contexts for numba_dpex's dpjit decorator.
"""

from numba.core.base import BaseContext
from numba.core.target_extension import CPU, target_registry


class Dpex(CPU):
    pass


DPEX_TARGET_NAME = "dpex"

# Register a target hierarchy token in Numba's target registry, this
# permits lookup and reference in user space by the string "dpex"
target_registry[DPEX_TARGET_NAME] = Dpex


class DpexTargetContext(BaseContext):
    def __init__(self, typingctx, target=DPEX_TARGET_NAME):
        super().__init__(typingctx, target)
