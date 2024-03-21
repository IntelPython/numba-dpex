# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_dpex import experimental as dpex_exp
from numba_dpex.kernel_api import Item


def _kernel(item: Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_compilation_mode_option_user_definition():
    with pytest.warns(UserWarning):
        dpex_exp.kernel(_compilation_mode="kernel")(_kernel)
