# SPDX-FileCopyrightText: 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests if dpex target overloads are not available at numba.njit and only
available at numba_dpex.dpjit.
"""

import pytest
from numba import njit, types
from numba.core import errors
from numba.extending import intrinsic, overload

from numba_dpex import dpjit
from numba_dpex.core.targets.dpjit_target import DPEX_TARGET_NAME


def foo():
    return 1


@overload(foo, target=DPEX_TARGET_NAME)
def ol_foo():
    return lambda: 1


@intrinsic(target=DPEX_TARGET_NAME)
def intrinsic_foo(
    ty_context,
):
    """A numba "intrinsic" function to inject dpctl.SyclEvent constructor code.

    Args:
        ty_context (numba.core.typing.context.Context): The typing context
            for the codegen.

    Returns:
        tuple(numba.core.typing.templates.Signature, function): A tuple of
            numba function signature type and a function object.
    """

    sig = types.int32(types.void)

    def codegen(context, builder, sig, args: list):
        return context.get_constant(types.int32, 1)

    return sig, codegen


def bar():
    return foo()


def intrinsic_bar():
    res = intrinsic_foo()
    return res


def test_dpex_overload_from_njit():
    bar_njit = njit(bar)

    with pytest.raises(errors.TypingError):
        bar_njit()


def test_dpex_overload_from_dpjit():
    bar_dpjit = dpjit(bar)
    bar_dpjit()


def test_dpex_intrinsic_from_njit():
    bar_njit = njit(intrinsic_bar)

    with pytest.raises(errors.TypingError):
        bar_njit()


def test_dpex_intrinsic_from_dpjit():
    bar_dpjit = dpjit(intrinsic_bar)
    bar_dpjit()
