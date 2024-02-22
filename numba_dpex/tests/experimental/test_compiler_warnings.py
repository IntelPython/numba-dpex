# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import pytest
from numba.core import types

from numba_dpex import DpctlSyclQueue, DpnpNdArray
from numba_dpex import experimental as dpex_exp
from numba_dpex import int64
from numba_dpex.core.types.kernel_api.index_space_ids import ItemType
from numba_dpex.kernel_api import Item


def _kernel(item: Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_compilation_mode_option_user_definition():
    with pytest.warns(UserWarning):
        dpex_exp.kernel(_compilation_mode="kernel")(_kernel)


def test_inline_threshold_level_warning():
    """
    Test compiler warning generation with an inline_threshold value of 3.
    """

    with pytest.warns(UserWarning):
        queue_ty = DpctlSyclQueue(dpctl.SyclQueue())
        i64arr_ty = DpnpNdArray(ndim=1, dtype=int64, layout="C", queue=queue_ty)
        kernel_sig = types.void(ItemType(1), i64arr_ty, i64arr_ty, i64arr_ty)
        dpex_exp.kernel(inline_threshold=3)(_kernel).compile(kernel_sig)
