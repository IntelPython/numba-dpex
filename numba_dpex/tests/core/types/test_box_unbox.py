# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for boxing and unboxing of types supported inside dpjit
"""

import functools
import sys

import dpctl
import dpctl.tensor as dpt
import dpnp
import numpy
import pytest

import numba_dpex as dpex

constructors_with_parent = [
    dpctl.SyclEvent,
    dpctl.SyclQueue,
    # numpy test is actually not related to the dpex, but helps make sure we are
    # doing it in the right way. Testing tests =)
    lambda: numpy.empty(10, dtype=numpy.float32),
    lambda: dpnp.empty(10, dtype=dpnp.float32),
    lambda: dpt.empty(10, dtype=dpt.float32),
]

ranges = [(10,), (10, 10), (10, 10, 10)]

constructors_without_parent = [
    functools.partial(dpex.Range, *r) for r in ranges
] + [
    functools.partial(dpex.NdRange, dpex.Range(*r), dpex.Range(*r))
    for r in ranges
]


@pytest.fixture(params=constructors_with_parent + constructors_without_parent)
def constructor(request):
    return request.param


@pytest.fixture(params=constructors_with_parent)
def constructor_wp(request):
    return request.param


@pytest.fixture(params=constructors_without_parent)
def constructor_np(request):
    return request.param


@dpex.dpjit
def unbox_box(a):
    return a


@dpex.dpjit
def unbox(a):
    return None


def test_unboxing_boxing_with_parent_and_return(constructor_wp):
    """Tests basic boxing and unboxing of an object.

    Checks if we can pass in and return the object to and from a dpjit decorated
    function. Checks if we have expected count of references.
    """

    e = constructor_wp()

    ref_before = sys.getrefcount(e)
    o = unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before + 1 == ref_after  # we have 'o' referencing same object
    assert id(o) == id(e)


def test_unboxing_boxing_with_return(constructor_np):
    """Tests basic boxing and unboxing of an object.

    Checks if we can pass in and return the object to and from a dpjit decorated
    function. Checks if we have expected count of references.
    """

    e = constructor_np()

    ref_before = sys.getrefcount(e)
    o = unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert id(o) != id(e)
    assert ref_before == ref_after  # we have 'o' referencing same object


def test_unboxing_boxing_with_same_assign(constructor):
    """Tests basic boxing and unboxing of an object.

    Checks if we can pass in and return the object to and from a dpjit decorated
    function. Checks if we have expected count of references.
    """

    e = constructor()

    ref_before = sys.getrefcount(e)
    e = unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after


def test_unboxing_boxing_without_return(constructor):
    """Tests basic boxing and unboxing of an object.

    Checks if we can pass in and return the object to and from a dpjit decorated
    function without assigning it to new variable.
    Checks if it does not affect reference count.
    """

    e = constructor()

    ref_before = sys.getrefcount(e)
    unbox_box(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after


def test_unboxing(constructor):
    """Tests basic unboxing of an object.

    Checks if we have expected amount of reference if we pass the object to a
    dpjit decorated function.
    """

    e = constructor()

    ref_before = sys.getrefcount(e)
    unbox(e)
    ref_after = sys.getrefcount(e)

    assert ref_before == ref_after
