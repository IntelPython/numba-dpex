################################################################################
#                                 Numba-DPPY
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import dpctl
import numpy as np
import pytest
from numba import njit

from numba_dppy.tests._helper import dpnp_debug, filter_strings

from ._helper import wrapper_function
from .dpnp_skip_test import skip_no_dpnp

# dpnp throws -30 (CL_INVALID_VALUE) when invoked with multiple kinds of
# devices at runtime, so testing for level_zero only
def skip(filter_str):
    if (filter_str == "opencl:gpu:0") or (filter_str == "opencl:cpu:0"):
        pytest.skip("CL_INVALID_VALUE")


list_of_size = [
    9,
    (2, 5),
    (3, 2, 4),
]

none_size = [None]


@pytest.fixture(params=list_of_size)
def unary_size(request):
    return request.param


@pytest.fixture(params=list_of_size + none_size)
def three_arg_size(request):
    return request.param


list_of_one_arg = [
    ("random_sample", 0.0, 1.0),
    ("ranf", 0.0, 1.0),
    ("sample", 0.0, 1.0),
    ("random", 0.0, 1.0),
    ("standard_exponential", 0.0, None),
    ("standard_normal", None, None),
    ("standard_cauchy", None, None),
]


@pytest.fixture(params=list_of_one_arg)
def one_arg_fn(request):
    function = wrapper_function(
        "size", f"np.random.{request.param[0]}(size)", globals()
    )
    return function, request.param


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_one_arg_fn(filter_str, one_arg_fn, unary_size, capfd):
    skip(filter_str)
    op, params = one_arg_fn
    name, low, high = params
    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(unary_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low is not None:
            assert np.all(actual >= low)
        if high is not None:
            assert np.all(actual < high)


list_of_two_arg_fn = [
    ("chisquare", 3, 0, None),
    ("exponential", 3.0, 0, None),
    ("gamma", 2.0, 0, None),
    ("geometric", 0.35, 0, None),
    ("poisson", 5.0, 0, None),
    ("rayleigh", 2.0, 0, None),
    ("standard_gamma", 2.0, 0, None),
    ("weibull", 5.0, 0, None),
]


@pytest.fixture(params=list_of_two_arg_fn)
def two_arg_fn(request):
    return request.param


def get_two_arg_fn(op_name):
    return wrapper_function("a, b", f"np.random.{op_name}(a, b)", globals())


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_two_arg_fn(filter_str, two_arg_fn, unary_size, capfd):
    skip(filter_str)
    op_name, first_arg, low, high = two_arg_fn

    if op_name == "gamma":
        pytest.skip(
            "AttributeError: 'NoneType' object has no attribute 'ravel'"
        )
    op = get_two_arg_fn(op_name)
    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(first_arg, unary_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low is not None and high is None:
            if np.isscalar(actual):
                assert actual >= low
            else:
                actual = actual.ravel()
                assert np.all(actual >= low)


list_of_three_arg_fn = [
    ("randint", 2, 23, 0, None),
    ("random_integers", 2, 23, 1, None),
    ("beta", 2.56, 0.8, 0, 1.0),
    ("binomial", 5, 0.0, 0, 1.0),
    ("gumbel", 0.5, 0.1, None, None),
    ("laplace", 0.0, 1.0, None, None),
    ("lognormal", 3.0, 1.0, None, None),
    ("multinomial", 100, np.array([1 / 7.0] * 5), 0, 100),
    ("multivariate_normal", (1, 2), [[1, 0], [0, 1]], None, None),
    ("negative_binomial", 1, 0.1, 0, None),
    ("normal", 0.0, 0.1, None, None),
    ("uniform", -1.0, 0.0, -1.0, 0.0),
]


@pytest.fixture(params=list_of_three_arg_fn)
def three_arg_fn(request):
    return request.param


def get_three_arg_fn(op_name):
    return wrapper_function(
        "a, b, c", f"np.random.{op_name}(a, b, c)", globals()
    )


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_three_arg_fn(filter_str, three_arg_fn, three_arg_size, capfd):
    skip(filter_str)
    op_name, first_arg, second_arg, low, high = three_arg_fn

    if op_name == "multinomial":
        pytest.skip("DPNP RNG Error: dpnp_rng_multinomial_c() failed")
    elif op_name == "multivariate_normal":
        pytest.skip(
            "No implementation of function Function(<class "
            "'numba_dppy.dpnp_iface.stubs.dpnp.multivariate_normal'>) found for signature"
        )
    elif op_name == "negative_binomial":
        pytest.skip("DPNP RNG Error: dpnp_rng_negative_binomial_c() failed.")
    elif op_name == "gumbel":
        pytest.skip("DPNP error")

    op = get_three_arg_fn(op_name)
    f = njit(op)
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(first_arg, second_arg, three_arg_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low is not None and high is None:
            if second_arg:
                low = first_arg
                high = second_arg
                assert np.all(actual >= low)
                assert np.all(actual <= high)
            else:
                high = first_arg
                assert np.all(actual >= low)
                assert np.all(actual <= high)
        elif low is not None and high is not None:
            if np.isscalar(actual):
                assert actual >= low
                assert actual <= high
            else:
                actual = actual.ravel()
                assert np.all(actual >= low)
                assert np.all(actual <= high)


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_rand(filter_str):
    skip(filter_str)

    @njit
    def f():
        c = np.random.rand(3, 2)
        return c

    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f()

        actual = actual.ravel()
        assert np.all(actual >= 0.0)
        assert np.all(actual < 1.0)


@skip_no_dpnp
@pytest.mark.parametrize("filter_str", filter_strings)
def test_hypergeometric(filter_str, three_arg_size):
    skip(filter_str)

    @njit
    def f(ngood, nbad, nsamp, size):
        res = np.random.hypergeometric(ngood, nbad, nsamp, size)
        return res

    ngood, nbad, nsamp = 100, 2, 10
    device = dpctl.SyclDevice(filter_str)
    with dpctl.device_context(device), dpnp_debug():
        actual = f(ngood, nbad, nsamp, three_arg_size)

        if np.isscalar(actual):
            assert actual >= 0
            assert actual <= min(nsamp, ngood + nbad)
        else:
            actual = actual.ravel()
            assert np.all(actual >= 0)
            assert np.all(actual <= min(nsamp, ngood + nbad))
