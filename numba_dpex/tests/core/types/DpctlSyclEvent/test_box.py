# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing and allocating for dpctl.SyclEvent
"""

import sys

from dpctl import SyclEvent

from numba_dpex import dpjit


def test_dpjit_constructor():
    """Test event delete that does not have parent"""

    @dpjit
    def func() -> SyclEvent:
        SyclEvent()
        return None

    # We just want to make sure execution did not crush. There are currently
    # no way to check if event wast destroyed, except manual run with debug
    # logs on.
    func()


def test_boxing_without_parent():
    """Test unboxing of the event that does not have parent"""

    @dpjit
    def func() -> SyclEvent:
        event = SyclEvent()
        return event

    e: SyclEvent = func()
    ref_cnt = sys.getrefcount(e)

    assert isinstance(e, SyclEvent)
    assert ref_cnt == 2
