# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

import numba_dpex as dpex
from numba_dpex import experimental as dpex_exp


def test_compilation_mode_option_user_definition():
    def kernel_func(a, b, c):
        i = dpex.get_global_id(0)
        c[i] = a[i] + b[i]

    with pytest.warns(warnings.warn(UserWarning)):
        dpex_exp.kernel(_compilation_mode="kernel")(kernel_func)
