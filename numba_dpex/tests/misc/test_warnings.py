# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dpnp
import pytest

import numba_dpex as dpex
from numba_dpex.core import config


@dpex.kernel
def foo(item, a):
    a[item.get_id(0)] = 0


def test_inline_threshold_negative_val_warning_():
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = -1

    with pytest.warns(UserWarning):
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp


def test_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))
