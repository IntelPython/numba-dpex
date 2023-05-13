# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for class DpexTargetContext."""


import dpctl
import dpnp
import pytest
from numba.core import typing
from numba.core.codegen import JITCPUCodegen

from numba_dpex import dpjit
from numba_dpex.core.targets.dpjit_target import DpexTargetContext

ctx = typing.Context()
dpexctx = DpexTargetContext(ctx)


def test_dpjit_target():
    assert dpexctx.lower_extensions == {}
    assert dpexctx.is32bit is False
    assert dpexctx.dpexrt is not None
    assert (
        isinstance(dpexctx._internal_codegen, type(JITCPUCodegen("numba.exec")))
        == 1
    )


def test_dpjit_target_refresh():
    try:
        dpexctx.refresh
    except KeyError:
        pytest.fail("Unexpected KeyError in dpjit_target.")


def test_dpjit_target_caching():
    @dpjit(cache=True)
    def func(a):
        b = dpnp.ones(10)
        return a + b

    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    b = dpnp.zeros(10, sycl_queue=q1)
    c = func(b)
    print(c)

    b = dpnp.ones(10, sycl_queue=q2)
    c = func(b)
    print(c)

    b = dpnp.ones(5)
    c = func(b)
    print(c)

    b = dpnp.ones(10, sycl_queue=q2, dtype=dpnp.int32)
    c = func(b)
    print(c)
