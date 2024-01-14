# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for class DpexTargetContext."""


import pytest
from numba.core import typing
from numba.core.codegen import JITCPUCodegen

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
