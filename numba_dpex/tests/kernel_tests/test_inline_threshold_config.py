# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from numba.core import compiler

import numba_dpex as dpex
from numba_dpex.kernel_api import Item


def kernel_func(item: Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]


def test_inline_threshold_set_using_config():
    oldConfig = dpex.config.INLINE_THRESHOLD
    dpex.config.INLINE_THRESHOLD = None

    disp = dpex.kernel(kernel_func)
    flags = compiler.Flags()
    disp.targetdescr.options.parse_as_flags(flags, disp.targetoptions)

    assert flags.inline_threshold == 0

    dpex.config.INLINE_THRESHOLD = 2

    flags = compiler.Flags()
    disp.targetdescr.options.parse_as_flags(flags, disp.targetoptions)

    assert flags.inline_threshold == 2

    dpex.config.INLINE_THRESHOLD = oldConfig


def test_inline_threshold_set_using_decorator_option():
    """
    Test setting the inline_threshold value using the kernel decorator flag
    """

    disp = dpex.kernel(inline_threshold=2)(kernel_func)
    flags = compiler.Flags()
    disp.targetdescr.options.parse_as_flags(flags, disp.targetoptions)

    assert flags.inline_threshold == 2


def test_inline_threshold_set_using_decorator_supersedes_config_option():
    oldConfig = dpex.config.INLINE_THRESHOLD
    dpex.config.INLINE_THRESHOLD = None

    disp = dpex.kernel(inline_threshold=3)(kernel_func)
    flags = compiler.Flags()
    disp.targetdescr.options.parse_as_flags(flags, disp.targetoptions)

    print(flags.inline_threshold)
    assert flags.inline_threshold == 3

    dpex.config.INLINE_THRESHOLD = oldConfig
