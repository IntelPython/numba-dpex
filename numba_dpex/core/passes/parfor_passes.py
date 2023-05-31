# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


from numba.core.compiler_machinery import register_pass
from numba.core.target_extension import target_override
from numba.core.typed_passes import PreParforPass


@register_pass(mutates_CFG=True, analysis_only=False)
class PreParforPass(PreParforPass):
    """
    A wrapper around Numba's PreParforPass that runs the pass inside a
    dpex target context.

    The target_override context manager ensures that any function that is
    compiled inside the Numba PreParforPass uses the DpexTargetContext.
    """

    _name = "dpex_pre_parfor_pass"

    def run_pass(self, state):
        with target_override("dpex"):
            super().run_pass(state)

        return True
