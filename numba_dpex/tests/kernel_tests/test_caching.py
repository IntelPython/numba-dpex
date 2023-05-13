# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import string

import dpctl.tensor as dpt
import numpy as np
import pytest

import numba_dpex as dpex
from numba_dpex.core.caching import LRUCache
from numba_dpex.core.kernel_interface.dispatcher import JitKernel
from numba_dpex.tests._helper import filter_strings


def test_LRUcache_operations():
    """Test rigorous caching operations.

    Performs different permutations of caching operations
    and check if the state of the cache is correct.
    """

    alphabet = list(string.ascii_lowercase)
    cache = LRUCache(name="testcache", capacity=4, pyfunc=None)
    assert str(cache) == "{}" and cache.head is None and cache.tail is None

    states = []
    for i in range(4):
        cache.put(i, alphabet[i])
        tail_key = cache.get(cache.tail.key)
        head_key = cache.get(cache.head.key)
        states.append((cache, tail_key, head_key))
    assert (
        str(states)
        == "["
        + "({(2: c), (1: b), (3: d), (0: a)}, 'a', 'a'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'b', 'a'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'c', 'b'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'd', 'a')"
        + "]"
    )

    states = []
    picking_order = [3, 1, 0, 2, 2]
    for index in picking_order:
        value = cache.get(index)
        states.append((value, cache, cache.head, cache.tail))
    assert (
        str(states)
        == "["
        + "('d', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (3: d)), "
        + "('b', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (1: b)), "
        + "('a', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (0: a)), "
        + "('c', {(3: d), (1: b), (0: a), (2: c)}, (3: d), (2: c)), "
        + "('c', {(3: d), (1: b), (0: a), (2: c)}, (3: d), (2: c))"
        + "]"
    )

    states = []
    for i in range(5, 10):
        cache.put(i, alphabet[i])
        tail_key = cache.get(cache.tail.key)
        head_key = cache.get(cache.head.key)
        states.append((cache, tail_key, head_key))
    assert (
        str(states)
        == "["
        + "({(8: i), (2: c), (9: j), (1: b)}, 'f', 'b'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'g', 'c'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'h', 'b'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'i', 'c'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'j', 'b')"
        + "]"
    )
    assert str(cache.evicted) == "{3: 'd', 0: 'a', 5: 'f', 6: 'g', 7: 'h'}"

    picking_order = [2, 1, 3]
    states = []
    for index in picking_order:
        value = cache.get(index)
        states.append((value, cache, cache.head, cache.tail))
    assert (
        str(states)
        == "["
        + "('c', {(9: j), (2: c), (1: b), (3: d)}, (8: i), (2: c)), "
        + "('b', {(9: j), (2: c), (1: b), (3: d)}, (8: i), (1: b)), "
        + "('d', {(9: j), (2: c), (1: b), (3: d)}, (9: j), (3: d))"
        + "]"
    )
    assert str(cache.evicted) == "{0: 'a', 5: 'f', 6: 'g', 7: 'h', 8: 'i'}"

    cache.put(0, "x")
    assert (
        str(cache) == "{(2: c), (1: b), (3: d), (0: x)}"
        and str(cache.head) == "(2: c)"
        and str(cache.tail) == "(0: x)"
    )
    assert str(cache.evicted) == "{5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}"

    cache.put(6, "y")
    assert (
        str(cache) == "{(1: b), (3: d), (0: x), (6: y)}"
        and str(cache.head) == "(1: b)"
        and str(cache.tail) == "(6: y)"
    )
    assert str(cache.evicted) == "{5: 'f', 7: 'h', 8: 'i', 9: 'j', 2: 'c'}"


@pytest.mark.parametrize("filter_str", filter_strings)
def test_caching_hit_counts(filter_str):
    """Tests the correct number of cache hits.
    If a Dispatcher is invoked 10 times and if the caching is enabled,
    then the total number of cache hits will be 9. Given the fact that
    the first time the kernel will be compiled and it will be loaded
    off the cache for the next time on.

    Args:
        filter_str (str): The device name coming from filter_strings in
        ._helper.py
    """

    def data_parallel_sum(x, y, z):
        """
        Vector addition using the ``kernel`` decorator.
        """
        i = dpex.get_global_id(0)
        z[i] = x[i] + y[i]

    a = dpt.arange(0, 100, device=filter_str)
    b = dpt.arange(0, 100, device=filter_str)
    c = dpt.zeros_like(a, device=filter_str)

    expected = dpt.asnumpy(a) + dpt.asnumpy(b)

    d = JitKernel(data_parallel_sum)

    d_launcher = d[100]

    N = 10
    for i in range(N):
        d_launcher(a, b, c)
    actual = dpt.asnumpy(c)

    assert np.array_equal(expected, actual) and (d_launcher.cache_hits == N - 1)


def load_from_index_file(hashtable, index_file):
    """Loads values from the index file

    Loads all the values mapped by 'hashtable'
    from an 'index_file' object.
    """
    evicted = {}
    for k in hashtable.keys():
        evicted[k] = index_file.load(k)
    return evicted


def test_LRUcache_with_index_file_operations():
    """Test rigorous caching operations.

    Performs different permutations of caching operations
    and check if the state of the cache is correct.

    This is similar to test_LRUcache_operations() but the
    evicted items are stored in a file and we check if the
    items are stored correctly.
    """

    def func(a):
        return a + 1

    alphabet = list(string.ascii_lowercase)
    cache = LRUCache(name="testcache", capacity=4, pyfunc=func)
    assert str(cache) == "{}" and cache.head is None and cache.tail is None

    states = []
    for i in range(4):
        cache.put(i, alphabet[i])
        tail_key = cache.get(cache.tail.key)
        head_key = cache.get(cache.head.key)
        states.append((cache, tail_key, head_key))
    assert (
        str(states)
        == "["
        + "({(2: c), (1: b), (3: d), (0: a)}, 'a', 'a'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'b', 'a'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'c', 'b'), "
        + "({(2: c), (1: b), (3: d), (0: a)}, 'd', 'a')"
        + "]"
    )

    states = []
    picking_order = [3, 1, 0, 2, 2]
    for index in picking_order:
        value = cache.get(index)
        states.append((value, cache, cache.head, cache.tail))
    assert (
        str(states)
        == "["
        + "('d', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (3: d)), "
        + "('b', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (1: b)), "
        + "('a', {(3: d), (1: b), (0: a), (2: c)}, (2: c), (0: a)), "
        + "('c', {(3: d), (1: b), (0: a), (2: c)}, (3: d), (2: c)), "
        + "('c', {(3: d), (1: b), (0: a), (2: c)}, (3: d), (2: c))"
        + "]"
    )

    states = []
    for i in range(5, 10):
        cache.put(i, alphabet[i])
        tail_key = cache.get(cache.tail.key)
        head_key = cache.get(cache.head.key)
        states.append((cache, tail_key, head_key))
    assert (
        str(states)
        == "["
        + "({(8: i), (2: c), (9: j), (1: b)}, 'f', 'b'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'g', 'c'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'h', 'b'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'i', 'c'), "
        + "({(8: i), (2: c), (9: j), (1: b)}, 'j', 'b')"
        + "]"
    )
    assert (
        str(load_from_index_file(cache.evicted, cache._cache_file))
        == "{3: 'd', 0: 'a', 5: 'f', 6: 'g', 7: 'h'}"
    )

    picking_order = [2, 1, 3]
    states = []
    for index in picking_order:
        value = cache.get(index)
        states.append((value, cache, cache.head, cache.tail))
    assert (
        str(states)
        == "["
        + "('c', {(9: j), (2: c), (1: b), (3: d)}, (8: i), (2: c)), "
        + "('b', {(9: j), (2: c), (1: b), (3: d)}, (8: i), (1: b)), "
        + "('d', {(9: j), (2: c), (1: b), (3: d)}, (9: j), (3: d))"
        + "]"
    )
    assert (
        str(load_from_index_file(cache.evicted, cache._cache_file))
        == "{0: 'a', 5: 'f', 6: 'g', 7: 'h', 8: 'i'}"
    )

    cache.put(0, "x")
    assert (
        str(cache) == "{(2: c), (1: b), (3: d), (0: x)}"
        and str(cache.head) == "(2: c)"
        and str(cache.tail) == "(0: x)"
    )
    assert (
        str(load_from_index_file(cache.evicted, cache._cache_file))
        == "{5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j'}"
    )

    cache.put(6, "y")
    assert (
        str(cache) == "{(1: b), (3: d), (0: x), (6: y)}"
        and str(cache.head) == "(1: b)"
        and str(cache.tail) == "(6: y)"
    )
    assert (
        str(load_from_index_file(cache.evicted, cache._cache_file))
        == "{5: 'f', 7: 'h', 8: 'i', 9: 'j', 2: 'c'}"
    )
