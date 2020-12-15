import numba
import numpy
import sys

from numba_dppy.testing import DPPYTestCase

import dpctl.dptensor.numpy_usm_shared as usmarray


def p1(a):
    return a * 2.0 + 13


f1 = numba.njit(p1)


@numba.njit()
def f2(a):
    return a


@numba.njit()
def f3(a, b):  # a is usmarray, b is numpy
    return a * usmarray.asarray(b)


@numba.njit()
def f4():
    return usmarray.ones(10)


def p5(a, b):  # a is usmarray, b is numpy
    return a * b


f5 = numba.njit(p5)


@numba.njit()
def f6(a):
    return a + 13


@numba.njit()
def f7(a):  # a is usmarray
    # implicit conversion of a to numpy.ndarray
    b = numpy.ones(10)
    c = a * b
    d = a.argsort()  # with no implicit conversion this fails


@numba.njit
def f8(a):
    return usmarray.as_ndarray(a)


@numba.njit
def f9(a):
    return usmarray.from_ndarray(a)


@numba.njit
def f10():
    return usmarray.empty((10, 10))


@numba.njit
def f11(x):
    return x.shape


@numba.njit
def f12(x):
    return x.T


# --------------------------------------------------------------------------------


class TestUsmArray(DPPYTestCase):
    def setUp(self):
        self.z1 = numpy.ones(10)
        self.a = usmarray.ones(10)
        self.nd2 = usmarray.as_ndarray(self.a)
        self.c = self.a * 5
        self.nd3 = f8(self.a)
        self.dp3 = f10()

    def test_python_numpy(self):
        print("------------------- Testing Python Numpy")
        sys.stdout.flush()
        z2 = p1(self.z1)
        print("z2:", z2, type(z2))
        assert type(z2) == numpy.ndarray

    def test_numba_numpy(self):
        print("------------------- Testing Numba Numpy")
        sys.stdout.flush()
        z2 = f1(self.z1)
        print("z2:", z2, type(z2))
        assert type(z2) == numpy.ndarray

    def test_usmarray_ones(self):
        print("------------------- Testing usmarray ones")
        sys.stdout.flush()
        print("a:", self.a, type(self.a))
        assert isinstance(self.a, usmarray.ndarray)
        assert usmarray.has_array_interface(self.a)

    def test_usmarray_usmarray_as_ndarray(self):
        print("------------------- Testing usmarray.usmarray.as_ndarray")
        sys.stdout.flush()
        nd1 = self.a.as_ndarray()
        print("nd1:", nd1, type(nd1))
        assert type(nd1) == numpy.ndarray

    def test_usmarray_as_ndarray(self):
        print("------------------- Testing usmarray.as_ndarray")
        sys.stdout.flush()
        nd2 = usmarray.as_ndarray(self.a)
        print("nd2:", nd2, type(nd2))
        assert type(nd2) == numpy.ndarray

    def test_usmarray_from_ndarray(self):
        print("------------------- Testing usmarray.from_ndarray")
        sys.stdout.flush()
        dp1 = usmarray.from_ndarray(self.nd2)
        print("dp1:", dp1, type(dp1))
        assert isinstance(dp1, usmarray.ndarray)
        assert usmarray.has_array_interface(dp1)

    def test_usmarray_multiplication(self):
        print("------------------- Testing usmarray multiplication")
        sys.stdout.flush()
        c = self.a * 5
        print("c", c, type(c))
        assert isinstance(c, usmarray.ndarray)
        assert usmarray.has_array_interface(c)

    def test_python_usmarray(self):
        print("------------------- Testing Python usmarray")
        sys.stdout.flush()
        b = p1(self.c)
        print("b:", b, type(b))
        assert isinstance(b, usmarray.ndarray)
        assert usmarray.has_array_interface(b)
        del b

    def test_python_mixing_usmarray_and_numpy_ndarray(self):
        print("------------------- Testing Python mixing usmarray and numpy.ndarray")
        sys.stdout.flush()
        h = p5(self.a, self.z1)
        print("h:", h, type(h))
        assert isinstance(h, usmarray.ndarray)
        assert usmarray.has_array_interface(h)
        del h

    def test_numba_usmarray_2(self):
        print("------------------- Testing Numba usmarray 2")
        sys.stdout.flush()
        d = f2(self.a)
        print("d:", d, type(d))
        assert isinstance(d, usmarray.ndarray)
        assert usmarray.has_array_interface(d)
        del d

    def test_numba_usmarray(self):
        # fails if run after test_all

        print("------------------- Testing Numba usmarray")
        sys.stdout.flush()
        b = f1(self.c)
        print("b:", b, type(b))
        assert isinstance(b, usmarray.ndarray)
        assert usmarray.has_array_interface(b)
        del b

    def test_numba_usmarray_constructor_from_numpy_ndarray(self):
        """
        print("------------------- Testing Numba usmarray constructor from numpy.ndarray")
        sys.stdout.flush()
        e = f3(self.a, self.z1)
        print("e:", e, type(e))
        assert(isinstance(e, usmarray.ndarray))
        """

    def test_numba_mixing_usmarray_and_constant(self):
        print("------------------- Testing Numba mixing usmarray and constant")
        sys.stdout.flush()
        g = f6(self.a)
        print("g:", g, type(g))
        assert isinstance(g, usmarray.ndarray)
        assert usmarray.has_array_interface(g)
        del g

    def test_numba_mixing_usmarray_and_numpy_ndarray(self):
        print("------------------- Testing Numba mixing usmarray and numpy.ndarray")
        sys.stdout.flush()
        h = f5(self.a, self.z1)
        print("h:", h, type(h))
        assert isinstance(h, usmarray.ndarray)
        assert usmarray.has_array_interface(h)
        del h

    def test_numba_usmarray_functions(self):
        print("------------------- Testing Numba usmarray functions")
        sys.stdout.flush()
        f = f4()
        print("f:", f, type(f))
        assert isinstance(f, usmarray.ndarray)
        assert usmarray.has_array_interface(f)
        del f

    def test_numba_usmarray_as_ndarray(self):
        print("------------------- Testing Numba usmarray.as_ndarray")
        sys.stdout.flush()
        nd3 = f8(self.a)
        print("nd3:", nd3, type(nd3))
        assert type(nd3) == numpy.ndarray

    def test_numba_usmarray_from_ndarray(self):
        print("------------------- Testing Numba usmarray.from_ndarray")
        sys.stdout.flush()
        dp2 = f9(self.nd3)
        print("dp2:", dp2, type(dp2))
        assert isinstance(dp2, usmarray.ndarray)
        assert usmarray.has_array_interface(dp2)
        del self.nd3
        del dp2

    def test_numba_usmarray_empty(self):
        print("------------------- Testing Numba usmarray.empty")
        sys.stdout.flush()
        dp3 = f10()
        print("dp3:", dp3, type(dp3))
        assert isinstance(dp3, usmarray.ndarray)
        assert usmarray.has_array_interface(dp3)

    def test_numba_usmarray_shape(self):
        print("------------------- Testing Numba usmarray.shape")
        sys.stdout.flush()
        s1 = f11(self.dp3)
        print("s1:", s1, type(s1))

    def test_numba_usmarray_T(self):
        print("------------------- Testing Numba usmarray.T")
        sys.stdout.flush()
        dp4 = f12(self.dp3)
        print("dp4:", dp4, type(dp4))
        assert isinstance(dp4, usmarray.ndarray)
        assert usmarray.has_array_interface(dp4)
        del self.dp3
        del dp4
