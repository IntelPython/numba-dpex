# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import dpnp
import pytest

import numba_dpex as dpex
import numba_dpex.config as config


@dpex.kernel(enable_cache=False)
def foo(a):
    a[dpex.get_global_id(0)] = 0


def test_opt_warning():
    bkp = config.DPEX_OPT
    config.DPEX_OPT = 3

    with pytest.warns(UserWarning):
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))

    config.DPEX_OPT = bkp


def test_inline_threshold_eq_3_warning():
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = 3

    with pytest.warns(UserWarning):
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp


def test_inline_threshold_negative_val_warning_():
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = -1

    with pytest.warns(UserWarning):
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp


def test_inline_threshold_gt_3_warning():
    bkp = config.INLINE_THRESHOLD
    config.INLINE_THRESHOLD = 4

    with pytest.warns(UserWarning):
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))

    config.INLINE_THRESHOLD = bkp


def test_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dpex.call_kernel(foo, dpex.Range(10), dpnp.arange(10))
