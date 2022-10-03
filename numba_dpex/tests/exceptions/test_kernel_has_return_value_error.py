# SPDX-FileCopyrightText: 2022 Intel Corp.
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor as dpt
import pytest

import numba_dpex as dpex


@dpex.kernel
def illegal_kernel(a, b):
    i = dpex.get_global_id(0)
    return a[i] + b[i]


def test_kernel_has_return_value_error():

    a = dpt.arange(0, 100, 1)
    b = dpt.arange(0, 100, 1)
    with pytest.raises(dpex.core.dpex_exceptions.KernelHasReturnValueError):
        illegal_kernel(a, b)
