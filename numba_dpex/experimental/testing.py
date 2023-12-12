# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tools for testing, not intended for regular use."""


from numba.core import types
from numba.extending import intrinsic

from numba_dpex import dpjit
from numba_dpex.core.runtime.context import DpexRTContext


@intrinsic(target="cpu")
def _kernel_cache_size(
    typingctx,  # pylint: disable=W0613
):
    sig = types.int64()

    def codegen(ctx, builder, sig, llargs):  # pylint: disable=W0613
        dpexrt = DpexRTContext(ctx)
        return dpexrt.kernel_cache_size(builder)

    return sig, codegen


@dpjit
def kernel_cache_size() -> int:
    """Returns kernel cache size."""
    return _kernel_cache_size()  # pylint: disable=E1120
