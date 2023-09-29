# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing and unboxing of types supported inside dpjit
"""

import sys

import dpctl

from numba_dpex import dpjit


@dpjit
def unbox_box(a):
    return a


@dpjit
def unbox(a):
    return None


def test_unboxing_boxing_with_return():
    """Tests basic boxing and unboxing of a dpctl.SyclEvent object.

    Checks if we can pass in and return a dpctl.SyclEvent object to and
    from a dpjit decorated function. Checks if we have expected count of
    references.
    """

    e = dpctl.SyclEvent()

    ref_before = sys.getrefcount(e)
    o = unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before + 1 == ref_after  # we have 'o' referencing same object
    assert id(o) == id(e)


def test_unboxing_boxing_with_same_assign():
    """Tests basic boxing and unboxing of a dpctl.SyclEvent object.

    Checks if we can pass in and return a dpctl.SyclEvent object to and
    from a dpjit decorated function. Checks if we have expected count of
    references.
    """

    e = dpctl.SyclEvent()

    ref_before = sys.getrefcount(e)
    e = unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after


def test_unboxing_boxing_without_return():
    """Tests basic boxing and unboxing of a dpctl.SyclEvent object.

    Checks if we can pass in and return a dpctl.SyclEvent object to and
    from a dpjit decorated function without assigning it to new variable.
    Checks if it does not affect reference count.
    """

    e = dpctl.SyclEvent()

    ref_before = sys.getrefcount(e)
    unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after


def test_unboxing():
    """Tests basic unboxing of a dpctl.SyclEvent object.

    Checks if we have expected amount of reference if we pass dpctl.SyclEvent
    object to a dpjit decorated function.
    """

    e = dpctl.SyclEvent()

    ref_before = sys.getrefcount(e)
    unbox(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after
