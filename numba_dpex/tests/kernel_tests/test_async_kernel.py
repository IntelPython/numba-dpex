# SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpnp
import pytest
from numba.core import types
from numba.core.errors import TypingError
from numba.extending import intrinsic

import numba_dpex as dpex
from numba_dpex import dpjit
from numba_dpex.core.runtime.context import DpexRTContext
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME
from numba_dpex.kernel_api import Item, Range


@intrinsic(target=DPEX_TARGET_NAME)
def _kernel_cache_size(
    typingctx,  # pylint: disable=W0613
):
    sig = types.int64()

    def codegen(ctx, builder, sig, llargs):  # pylint: disable=W0613
        dpexrt = DpexRTContext(ctx)
        return dpexrt.kernel_cache_size(builder)

    return sig, codegen


@dpjit
def kernel_cache_size() -> int:
    """Returns kernel cache size."""
    return _kernel_cache_size()  # pylint: disable=E1120


@dpex.kernel(
    release_gil=False,
    no_compile=True,
    no_cpython_wrapper=True,
    no_cfunc_wrapper=True,
)
def add(item: Item, a, b, c):
    i = item.get_id(0)
    c[i] = b[i] + a[i]


def test_async_add():
    size = 10
    a = dpnp.ones(size)
    b = dpnp.ones(size)
    c = dpnp.zeros(size)

    r = Range(size)

    host_ref, event_ref = dpex.call_kernel_async(
        add,
        r,
        (),
        a,
        b,
        c,
    )

    assert isinstance(host_ref, dpctl.SyclEvent)
    assert isinstance(event_ref, dpctl.SyclEvent)
    assert host_ref is not None
    assert event_ref is not None

    event_ref.wait()
    host_ref.wait()

    d = a + b
    assert dpnp.array_equal(c, d)


def test_async_dependent_add_list_exception():
    """Checks either ValueError is triggered if list was passed instead of
    tuple for the dependent events."""
    size = 10

    # TODO: should capture ValueError, but numba captures it and generates
    # TypingError. ValueError is still readable there.
    with pytest.raises(TypingError):
        dpex.call_kernel_async(
            add,
            Range(size),
            [dpctl.SyclEvent()],
            dpnp.ones(size),
            dpnp.ones(size),
            dpnp.ones(size),
        )


def test_async_dependent_add():
    size = 10
    a = dpnp.ones(size)
    b = dpnp.ones(size)
    c = dpnp.zeros(size)

    r = Range(size)

    host_ref, event_ref = dpex.call_kernel_async(
        add,
        r,
        (),
        a,
        b,
        c,
    )

    host2_ref, event2_ref = dpex.call_kernel_async(
        add,
        r,
        (event_ref,),
        a,
        c,
        b,
    )

    event2_ref.wait()
    d = dpnp.ones(size) * 3
    assert dpnp.array_equal(b, d)

    host_ref.wait()
    host2_ref.wait()


def test_async_add_from_cache():
    test_async_add()  # compile
    old_size = kernel_cache_size()
    test_async_add()  # use from cache
    new_size = kernel_cache_size()

    assert new_size == old_size
