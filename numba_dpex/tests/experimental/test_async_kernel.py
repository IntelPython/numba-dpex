# SPDX-FileCopyrightText: 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp

import numba_dpex as dpex
import numba_dpex.experimental as exp_dpex
from numba_dpex import Range
from numba_dpex.experimental import testing


@exp_dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def add(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = b[i] + a[i]


def test_async_add():
    size = 10
    a = dpnp.ones(size)
    b = dpnp.ones(size)
    c = dpnp.zeros(size)

    r = Range(size)

    host_ref, event_ref = exp_dpex.call_kernel_async(
        add,
        r,
        a,
        b,
        c,
    )

    assert type(host_ref) == dpctl.SyclEvent
    assert type(event_ref) == dpctl.SyclEvent
    assert host_ref is not None
    assert event_ref is not None

    event_ref.wait()
    host_ref.wait()

    d = a + b
    assert dpnp.array_equal(c, d)


def test_async_add_from_cache():
    test_async_add()  # compile
    old_size = testing.kernel_cache_size()
    test_async_add()  # use from cache
    new_size = testing.kernel_cache_size()

    assert new_size == old_size
