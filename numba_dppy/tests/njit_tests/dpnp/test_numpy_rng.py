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

from numba_dppy.context import device_context
import numpy as np
from numba import njit
import pytest
from numba_dppy.testing import dpnp_debug
from .dpnp_skip_test import dpnp_skip_test as skip_test

# dpnp throws -30 (CL_INVALID_VALUE) when invoked with multiple kinds of
# devices at runtime, so testing for level0 only
list_of_filter_strs = [
    # "opencl:gpu:0",
    "level0:gpu:0",
    # "opencl:cpu:0",
]


@pytest.fixture(params=list_of_filter_strs)
def filter_str(request):
    return request.param


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
    func_str = "def fn(size):\n    return np.random." + request.param[0] + "(size)"
    ldict = {}
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    return fn, request.param


def test_one_arg_fn(filter_str, one_arg_fn, unary_size, capfd):
    if skip_test(filter_str):
        pytest.skip()

    op, params = one_arg_fn
    name, low, high = params
    f = njit(op)
    with device_context(filter_str), dpnp_debug():
        actual = f(unary_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low != None:
            assert np.all(actual >= low)
        if high != None:
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
    func_str = (
        "def fn(first_arg, second_arg):\n\treturn np.random."
        + op_name
        + "(first_arg, second_arg)"
    )
    ldict = {}
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    return fn


def test_two_arg_fn(filter_str, two_arg_fn, unary_size, capfd):
    if skip_test(filter_str):
        pytest.skip()

    op_name, first_arg, low, high = two_arg_fn

    if op_name == "gamma":
        pytest.skip("AttributeError: 'NoneType' object has no attribute 'ravel'")
    op = get_two_arg_fn(op_name)
    f = njit(op)
    with device_context(filter_str), dpnp_debug():
        actual = f(first_arg, unary_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low != None and high == None:
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
    func_str = (
        "def fn(first_arg, second_arg, third_arg):\n\treturn np.random."
        + op_name
        + "(first_arg, second_arg, third_arg)"
    )
    ldict = {}
    exec(func_str, globals(), ldict)
    fn = ldict["fn"]
    return fn


def test_three_arg_fn(filter_str, three_arg_fn, three_arg_size, capfd):
    if skip_test(filter_str):
        pytest.skip()

    op_name, first_arg, second_arg, low, high = three_arg_fn

    if op_name == "multinomial":
        pytest.skip("DPNP RNG Error: dpnp_rng_multinomial_c() failed")
    elif op_name == "multivariate_normal":
        pytest.skip(
            "No implementation of function Function(<class "
            "'numba_dppy.dpnp_glue.stubs.dpnp.multivariate_normal'>) found for signature"
        )
    elif op_name == "negative_binomial":
        pytest.skip("DPNP RNG Error: dpnp_rng_negative_binomial_c() failed.")
    elif op_name == "gumbel":
        pytest.skip("DPNP error")

    op = get_three_arg_fn(op_name)
    f = njit(op)
    with device_context(filter_str), dpnp_debug():
        actual = f(first_arg, second_arg, three_arg_size)
        captured = capfd.readouterr()
        assert "dpnp implementation" in captured.out

        if low != None and high == None:
            if second_arg:
                low = first_arg
                high = second_arg
                assert np.all(actual >= low)
                assert np.all(actual <= high)
            else:
                high = first_arg
                assert np.all(actual >= low)
                assert np.all(actual <= high)
        elif low != None and high != None:
            if np.isscalar(actual):
                assert actual >= low
                assert actual <= high
            else:
                actual = actual.ravel()
                assert np.all(actual >= low)
                assert np.all(actual <= high)


def test_rand(filter_str):
    if skip_test(filter_str):
        pytest.skip()

    @njit
    def f():
        c = np.random.rand(3, 2)
        return c

    with device_context(filter_str), dpnp_debug():
        actual = f()

        actual = actual.ravel()
        assert np.all(actual >= 0.0)
        assert np.all(actual < 1.0)


def test_hypergeometric(filter_str, three_arg_size):
    if skip_test(filter_str):
        pytest.skip()

    @njit
    def f(ngood, nbad, nsamp, size):
        res = np.random.hypergeometric(ngood, nbad, nsamp, size)
        return res

    ngood, nbad, nsamp = 100, 2, 10
    with device_context(filter_str), dpnp_debug():
        actual = f(ngood, nbad, nsamp, three_arg_size)

        if np.isscalar(actual):
            assert actual >= 0
            assert actual <= min(nsamp, ngood + nbad)
        else:
            actual = actual.ravel()
            assert np.all(actual >= 0)
            assert np.all(actual <= min(nsamp, ngood + nbad))
