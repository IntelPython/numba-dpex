# Copyright 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl.tensor.numpy_usm_shared as usmarray
import numba
import numpy
import pytest


@numba.njit()
def numba_mul_add(a):
    return a * 2.0 + 13


@numba.njit()
def numba_add_const(a):
    return a + 13


@numba.njit()
def numba_mul(a, b):  # a is usmarray, b is numpy
    return a * b


@numba.njit()
def numba_mul_usmarray_asarray(a, b):  # a is usmarray, b is numpy
    return a * usmarray.asarray(b)


@numba.njit
def numba_usmarray_as_ndarray(a):
    return usmarray.as_ndarray(a)


@numba.njit
def numba_usmarray_from_ndarray(a):
    return usmarray.from_ndarray(a)


@numba.njit()
def numba_usmarray_ones():
    return usmarray.ones(10)


@numba.njit
def numba_usmarray_empty():
    return usmarray.empty((10, 10))


@numba.njit()
def numba_identity(a):
    return a


@numba.njit
def numba_shape(x):
    return x.shape


@numba.njit
def numba_T(x):
    return x.T


@numba.njit
def numba_reshape(x):
    return x.reshape((4, 3))


class TestUsmArray:
    def ndarray(self):
        """Create NumPy array"""
        return numpy.ones(10)

    def usmarray(self):
        """Create dpCtl USM array"""
        return usmarray.ones(10)

    def test_python_numpy(self):
        """Testing Python Numpy"""
        z2 = numba_mul_add.py_func(self.ndarray())
        assert type(z2) == numpy.ndarray, z2

    def test_numba_numpy(self):
        """Testing Numba Numpy"""
        z2 = numba_mul_add(self.ndarray())
        assert type(z2) == numpy.ndarray, z2

    def test_usmarray_ones(self):
        """Testing usmarray ones"""
        a = usmarray.ones(10)
        assert isinstance(a, usmarray.ndarray), type(a)
        assert usmarray.has_array_interface(a)

    def test_usmarray_usmarray_as_ndarray(self):
        """Testing usmarray.usmarray.as_ndarray"""
        nd1 = self.usmarray().as_ndarray()
        assert type(nd1) == numpy.ndarray, nd1

    def test_usmarray_as_ndarray(self):
        """Testing usmarray.as_ndarray"""
        nd2 = usmarray.as_ndarray(self.usmarray())
        assert type(nd2) == numpy.ndarray, nd2

    def test_usmarray_from_ndarray(self):
        """Testing usmarray.from_ndarray"""
        nd2 = usmarray.as_ndarray(self.usmarray())
        dp1 = usmarray.from_ndarray(nd2)
        assert isinstance(dp1, usmarray.ndarray), type(dp1)
        assert usmarray.has_array_interface(dp1)

    def test_usmarray_multiplication(self):
        """Testing usmarray multiplication"""
        c = self.usmarray() * 5
        assert isinstance(c, usmarray.ndarray), type(c)
        assert usmarray.has_array_interface(c)

    def test_python_usmarray_mul_add(self):
        """Testing Python usmarray"""
        c = self.usmarray() * 5
        b = numba_mul_add.py_func(c)
        assert isinstance(b, usmarray.ndarray), type(b)
        assert usmarray.has_array_interface(b)

    @pytest.mark.skip(reason="Fails if run tests in bunch")
    def test_numba_usmarray_mul_add(self):
        """Testing Numba usmarray"""
        c = self.usmarray() * 5
        b = numba_mul_add(c)
        assert isinstance(b, usmarray.ndarray), type(b)
        assert usmarray.has_array_interface(b)

    def test_python_mixing_usmarray_and_numpy_ndarray(self):
        """Testing Python mixing usmarray and numpy.ndarray"""
        h = numba_mul.py_func(self.usmarray(), self.ndarray())
        assert isinstance(h, usmarray.ndarray), type(h)
        assert usmarray.has_array_interface(h)

    def test_numba_usmarray_2(self):
        """Testing Numba usmarray 2"""
        d = numba_identity(self.usmarray())
        assert isinstance(d, usmarray.ndarray), type(d)
        assert usmarray.has_array_interface(d)

    @pytest.mark.xfail
    def test_numba_usmarray_constructor_from_numpy_ndarray(self):
        """Testing Numba usmarray constructor from numpy.ndarray"""
        e = numba_mul_usmarray_asarray(self.usmarray(), self.ndarray())
        assert isinstance(e, usmarray.ndarray), type(e)

    def test_numba_mixing_usmarray_and_constant(self):
        """Testing Numba mixing usmarray and constant"""
        g = numba_add_const(self.usmarray())
        assert isinstance(g, usmarray.ndarray), type(g)
        assert usmarray.has_array_interface(g)

    def test_numba_mixing_usmarray_and_numpy_ndarray(self):
        """Testing Numba mixing usmarray and numpy.ndarray"""
        h = numba_mul(self.usmarray(), self.ndarray())
        assert isinstance(h, usmarray.ndarray), type(h)
        assert usmarray.has_array_interface(h)

    def test_numba_usmarray_functions(self):
        """Testing Numba usmarray functions"""
        f = numba_usmarray_ones()
        assert isinstance(f, usmarray.ndarray), type(f)
        assert usmarray.has_array_interface(f)

    def test_numba_usmarray_as_ndarray(self):
        """Testing Numba usmarray.as_ndarray"""
        nd3 = numba_usmarray_as_ndarray(self.usmarray())
        assert type(nd3) == numpy.ndarray, nd3

    def test_numba_usmarray_from_ndarray(self):
        """Testing Numba usmarray.from_ndarray"""
        nd3 = numba_usmarray_as_ndarray(self.usmarray())
        dp2 = numba_usmarray_from_ndarray(nd3)
        assert isinstance(dp2, usmarray.ndarray), type(dp2)
        assert usmarray.has_array_interface(dp2)

    def test_numba_usmarray_empty(self):
        """Testing Numba usmarray.empty"""
        dp3 = numba_usmarray_empty()
        assert isinstance(dp3, usmarray.ndarray), type(dp3)
        assert usmarray.has_array_interface(dp3)

    def test_numba_usmarray_shape(self):
        """Testing Numba usmarray.shape"""
        s1 = numba_shape(numba_usmarray_empty())
        assert isinstance(s1, tuple), type(s1)
        assert s1 == (10, 10)

    def test_numba_usmarray_T(self):
        """Testing Numba usmarray.T"""
        dp4 = numba_T(numba_usmarray_empty())
        assert isinstance(dp4, usmarray.ndarray), type(dp4)
        assert usmarray.has_array_interface(dp4)

    @pytest.mark.xfail
    def test_numba_usmarray_reshape(self):
        """Testing Numba usmarray.reshape()"""
        a = usmarray.ones(12)
        s1 = numba_reshape(a)
        assert isinstance(s1, usmarray.ndarray), type(s1)
        assert s1.shape == (4, 3)
