# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0


import dpnp
import numpy

import numba_dpex as dpex


@dpex.func
def f(a):
    """A simple kernel function with return.

    Args:
        a (int, float): Input scalar value.

    Returns:
        int, float: Output scalar value
    """
    b = a + 1
    return b


@dpex.func
def f_raise(a):
    """A kernel function with a 'raise' statement.

    Args:
        a (int, float): Input scalar value.

    Raises:
        ValueError: Raised if computed value is not 2

    Returns:
        int, float: Output scalar value
    """
    b = a + 1
    if b != 2:
        raise ValueError("b should be 2")
    else:
        return b


@dpex.func
def f_assert(a):
    """A kernel function with 'assert' statement.

    Args:
        a (int, float): Input scalar value.

    Returns:
        int, float: Output scalar value
    """
    b = a + 1
    assert b == 2
    return b


@dpex.kernel
def kf(a, b):
    """A simple kernel.

    This kernel applies function 'f()' on input array 'a'.

    Args:
        a (dpnp.dpnp_array.dpnp_array): Input DPNP array.
        b (dpnp.dpnp_array.dpnp_array): Input DPNP array.
    """
    i = dpex.get_global_id(0)
    b[i] = f(a[i])


@dpex.kernel
def kfr(a, b):
    """A kernel that calls a function with 'raise' statement.

    This kernel applies function 'f_raise()' on input array 'a'.
    'f_raise()' has a 'raise' statement inside it.

    Args:
        a (dpnp.dpnp_array.dpnp_array): Input DPNP array.
        b (dpnp.dpnp_array.dpnp_array): Input DPNP array.
    """
    i = dpex.get_global_id(0)
    b[i] = f_raise(a[i])


@dpex.kernel
def kfa(a, b):
    """A kernel that calls a function with 'assert' statement.

    This kernel applies function 'f_assert()' on input array 'a'.
    'f_assert()' has a 'assert' statement inside it.

    Args:
        a (dpnp.dpnp_array.dpnp_array): Input DPNP array.
        b (dpnp.dpnp_array.dpnp_array): Input DPNP array.
    """
    i = dpex.get_global_id(0)
    b[i] = f_assert(a[i])


@dpex.kernel
def k_raise(a, b):
    """A kernel with 'raise' statement.

    Args:
        a (dpnp.dpnp_array.dpnp_array): Input DPNP array.
        b (dpnp.dpnp_array.dpnp_array): Input DPNP array.

    Raises:
        ValueError: Raised if the computed value 'b[i]' is not 2.
    """
    i = dpex.get_global_id(0)
    b[i] = f(a[i])
    if b[i] != 2:
        raise ValueError("b[i] should be 2")


@dpex.kernel
def k_assert(a, b):
    """A kernel with 'assert' statement.

    Args:
        a (dpnp.dpnp_array.dpnp_array): Input DPNP array.
        b (dpnp.dpnp_array.dpnp_array): Input DPNP array.
    """
    i = dpex.get_global_id(0)
    b[i] = f(a[i])
    assert b[i] == 2


def test_basic_pass():
    """A basic test that passes."""
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    print(type(a))

    kf[dpex.Range(10)](a, b)
    nb = dpnp.asnumpy(b)
    assert numpy.all(nb == 2)


def test_raise_in_kernel():
    """Tests 'k_raise()' kernel."""
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    try:
        k_raise[dpex.Range(10)](a, b)
    except Exception as e:
        assert (
            "Python exceptions and asserts are unsupported in numba-dpex kernel."
            in str(e)
        )


def test_assert_in_kernel():
    """Tests 'k_assert()' kernel."""
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    try:
        k_assert[dpex.Range(10)](a, b)
    except Exception as e:
        assert (
            "Python exceptions and asserts are unsupported in numba-dpex kernel."
            in str(e)
        ) or "Unsupported constraint encountered" in str(e)


def test_raise_in_kernel_function():
    """Tests 'kfr()' kernel."""
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    try:
        kfr[dpex.Range(10)](a, b)
    except Exception as e:
        assert (
            "Python exceptions and asserts are unsupported in numba-dpex kernel."
            in str(e)
        )


def test_assert_in_kernel_function():
    """Tests 'kfa()' kernel."""
    a = dpnp.ones(10)
    b = dpnp.ones(10)

    try:
        kfa[dpex.Range(10)](a, b)
    except Exception as e:
        assert (
            "Python exceptions and asserts are unsupported in numba-dpex kernel."
            in str(e)
        ) or "Unsupported constraint encountered" in str(e)
