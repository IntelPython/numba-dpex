# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba
import numpy
import unittest
import pytest

import dpctl
import dpctl.tensor.numpy_usm_shared as usmarray


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


class TestUsmArray(unittest.TestCase):
    def ndarray(self):
        """Create NumPy array"""
        return numpy.ones(10)

    def usmarray(self):
        """Create dpCtl USM array"""
        return usmarray.ones(10)

    def test_python_numpy(self):
        """Testing Python Numpy"""
        z2 = numba_mul_add.py_func(self.ndarray())
        self.assertEqual(type(z2), numpy.ndarray, z2)

    def test_numba_numpy(self):
        """Testing Numba Numpy"""
        z2 = numba_mul_add(self.ndarray())
        self.assertEqual(type(z2), numpy.ndarray, z2)

    def test_usmarray_ones(self):
        """Testing usmarray ones"""
        a = usmarray.ones(10)
        self.assertIsInstance(a, usmarray.ndarray, type(a))
        self.assertTrue(usmarray.has_array_interface(a))

    def test_usmarray_usmarray_as_ndarray(self):
        """Testing usmarray.usmarray.as_ndarray"""
        nd1 = self.usmarray().as_ndarray()
        self.assertEqual(type(nd1), numpy.ndarray, nd1)

    def test_usmarray_as_ndarray(self):
        """Testing usmarray.as_ndarray"""
        nd2 = usmarray.as_ndarray(self.usmarray())
        self.assertEqual(type(nd2), numpy.ndarray, nd2)

    def test_usmarray_from_ndarray(self):
        """Testing usmarray.from_ndarray"""
        nd2 = usmarray.as_ndarray(self.usmarray())
        dp1 = usmarray.from_ndarray(nd2)
        self.assertIsInstance(dp1, usmarray.ndarray, type(dp1))
        self.assertTrue(usmarray.has_array_interface(dp1))

    def test_usmarray_multiplication(self):
        """Testing usmarray multiplication"""
        c = self.usmarray() * 5
        self.assertIsInstance(c, usmarray.ndarray, type(c))
        self.assertTrue(usmarray.has_array_interface(c))

    def test_python_usmarray_mul_add(self):
        """Testing Python usmarray"""
        c = self.usmarray() * 5
        b = numba_mul_add.py_func(c)
        self.assertIsInstance(b, usmarray.ndarray, type(b))
        self.assertTrue(usmarray.has_array_interface(b))

    def test_numba_usmarray_mul_add(self):
        """Testing Numba usmarray"""
        # fails if run tests in bunch
        c = self.usmarray() * 5
        b = numba_mul_add(c)
        self.assertIsInstance(b, usmarray.ndarray, type(b))
        self.assertTrue(usmarray.has_array_interface(b))

    def test_python_mixing_usmarray_and_numpy_ndarray(self):
        """Testing Python mixing usmarray and numpy.ndarray"""
        h = numba_mul.py_func(self.usmarray(), self.ndarray())
        self.assertIsInstance(h, usmarray.ndarray, type(h))
        self.assertTrue(usmarray.has_array_interface(h))

    def test_numba_usmarray_2(self):
        """Testing Numba usmarray 2"""
        d = numba_identity(self.usmarray())
        self.assertIsInstance(d, usmarray.ndarray, type(d))
        self.assertTrue(usmarray.has_array_interface(d))

    @unittest.expectedFailure
    def test_numba_usmarray_constructor_from_numpy_ndarray(self):
        """Testing Numba usmarray constructor from numpy.ndarray"""
        e = numba_mul_usmarray_asarray(self.usmarray(), self.ndarray())
        self.assertIsInstance(e, usmarray.ndarray, type(e))

    def test_numba_mixing_usmarray_and_constant(self):
        """Testing Numba mixing usmarray and constant"""
        g = numba_add_const(self.usmarray())
        self.assertIsInstance(g, usmarray.ndarray, type(g))
        self.assertTrue(usmarray.has_array_interface(g))

    def test_numba_mixing_usmarray_and_numpy_ndarray(self):
        """Testing Numba mixing usmarray and numpy.ndarray"""
        h = numba_mul(self.usmarray(), self.ndarray())
        self.assertIsInstance(h, usmarray.ndarray, type(h))
        self.assertTrue(usmarray.has_array_interface(h))

    def test_numba_usmarray_functions(self):
        """Testing Numba usmarray functions"""
        f = numba_usmarray_ones()
        self.assertIsInstance(f, usmarray.ndarray, type(f))
        self.assertTrue(usmarray.has_array_interface(f))

    def test_numba_usmarray_as_ndarray(self):
        """Testing Numba usmarray.as_ndarray"""
        nd3 = numba_usmarray_as_ndarray(self.usmarray())
        self.assertEqual(type(nd3), numpy.ndarray, nd3)

    def test_numba_usmarray_from_ndarray(self):
        """Testing Numba usmarray.from_ndarray"""
        nd3 = numba_usmarray_as_ndarray(self.usmarray())
        dp2 = numba_usmarray_from_ndarray(nd3)
        self.assertIsInstance(dp2, usmarray.ndarray, type(dp2))
        self.assertTrue(usmarray.has_array_interface(dp2))

    def test_numba_usmarray_empty(self):
        """Testing Numba usmarray.empty"""
        dp3 = numba_usmarray_empty()
        self.assertIsInstance(dp3, usmarray.ndarray, type(dp3))
        self.assertTrue(usmarray.has_array_interface(dp3))

    def test_numba_usmarray_shape(self):
        """Testing Numba usmarray.shape"""
        s1 = numba_shape(numba_usmarray_empty())
        self.assertIsInstance(s1, tuple, type(s1))
        self.assertEqual(s1, (10, 10))

    def test_numba_usmarray_T(self):
        """Testing Numba usmarray.T"""
        dp4 = numba_T(numba_usmarray_empty())
        self.assertIsInstance(dp4, usmarray.ndarray, type(dp4))
        self.assertTrue(usmarray.has_array_interface(dp4))

    @unittest.expectedFailure
    def test_numba_usmarray_reshape(self):
        """Testing Numba usmarray.reshape()"""
        a = usmarray.ones(12)
        s1 = numba_reshape(a)
        self.assertIsInstance(s1, usmarray.ndarray, type(s1))
        self.assertEqual(s1.shape, (4, 3))
