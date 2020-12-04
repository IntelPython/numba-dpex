from __future__ import print_function, division, absolute_import

import numba
import numba.dppl.dparray as dparray
import numpy
import sys


def p1(a):
    return a * 2.0 + 13


f1 = numba.njit(p1)


@numba.njit()
def f2(a):
    return a


@numba.njit()
def f3(a, b):  # a is dparray, b is numpy
    return a * dparray.asarray(b)


@numba.njit()
def f4():
    return dparray.ones(10)


def p5(a, b):  # a is dparray, b is numpy
    return a * b


f5 = numba.njit(p5)


@numba.njit()
def f6(a):
    return a + 13


@numba.njit()
def f7(a):  # a is dparray
    # implicit conversion of a to numpy.ndarray
    b = numpy.ones(10)
    c = a * b
    d = a.argsort()  # with no implicit conversion this fails


@numba.njit
def f8(a):
    return dparray.as_ndarray(a)


@numba.njit
def f9(a):
    return dparray.from_ndarray(a)


@numba.njit
def f10():
    return dparray.empty((10, 10))


@numba.njit
def f11(x):
    return x.shape


@numba.njit
def f12(x):
    return x.T


# --------------------------------------------------------------------------------

print("------------------- Testing Python Numpy")
sys.stdout.flush()
z1 = numpy.ones(10)
z2 = p1(z1)
print("z2:", z2, type(z2))
assert type(z2) == numpy.ndarray

print("------------------- Testing Numba Numpy")
sys.stdout.flush()
z1 = numpy.ones(10)
z2 = f1(z1)
print("z2:", z2, type(z2))
assert type(z2) == numpy.ndarray

print("------------------- Testing dparray ones")
sys.stdout.flush()
a = dparray.ones(10)
print("a:", a, type(a))
assert isinstance(a, dparray.ndarray)
assert dparray.has_array_interface(a)

print("------------------- Testing dparray.dparray.as_ndarray")
sys.stdout.flush()
nd1 = a.as_ndarray()
print("nd1:", nd1, type(nd1))
assert type(nd1) == numpy.ndarray

print("------------------- Testing dparray.as_ndarray")
sys.stdout.flush()
nd2 = dparray.as_ndarray(a)
print("nd2:", nd2, type(nd2))
assert type(nd2) == numpy.ndarray

print("------------------- Testing dparray.from_ndarray")
sys.stdout.flush()
dp1 = dparray.from_ndarray(nd2)
print("dp1:", dp1, type(dp1))
assert isinstance(dp1, dparray.ndarray)
assert dparray.has_array_interface(dp1)

print("------------------- Testing dparray multiplication")
sys.stdout.flush()
c = a * 5
print("c", c, type(c))
assert isinstance(c, dparray.ndarray)
assert dparray.has_array_interface(c)

print("------------------- Testing Python dparray")
sys.stdout.flush()
b = p1(c)
print("b:", b, type(b))
assert isinstance(b, dparray.ndarray)
assert dparray.has_array_interface(b)
del b

print("------------------- Testing Python mixing dparray and numpy.ndarray")
sys.stdout.flush()
h = p5(a, z1)
print("h:", h, type(h))
assert isinstance(h, dparray.ndarray)
assert dparray.has_array_interface(h)
del h

print("------------------- Testing Numba dparray 2")
sys.stdout.flush()
d = f2(a)
print("d:", d, type(d))
assert isinstance(d, dparray.ndarray)
assert dparray.has_array_interface(d)
del d

print("------------------- Testing Numba dparray")
sys.stdout.flush()
b = f1(c)
print("b:", b, type(b))
assert isinstance(b, dparray.ndarray)
assert dparray.has_array_interface(b)
del b

"""
print("------------------- Testing Numba dparray constructor from numpy.ndarray")
sys.stdout.flush()
e = f3(a, z1)
print("e:", e, type(e))
assert(isinstance(e, dparray.ndarray))
"""

print("------------------- Testing Numba mixing dparray and constant")
sys.stdout.flush()
g = f6(a)
print("g:", g, type(g))
assert isinstance(g, dparray.ndarray)
assert dparray.has_array_interface(g)
del g

print("------------------- Testing Numba mixing dparray and numpy.ndarray")
sys.stdout.flush()
h = f5(a, z1)
print("h:", h, type(h))
assert isinstance(h, dparray.ndarray)
assert dparray.has_array_interface(h)
del h

print("------------------- Testing Numba dparray functions")
sys.stdout.flush()
f = f4()
print("f:", f, type(f))
assert isinstance(f, dparray.ndarray)
assert dparray.has_array_interface(f)
del f

print("------------------- Testing Numba dparray.as_ndarray")
sys.stdout.flush()
nd3 = f8(a)
print("nd3:", nd3, type(nd3))
assert type(nd3) == numpy.ndarray

print("------------------- Testing Numba dparray.from_ndarray")
sys.stdout.flush()
dp2 = f9(nd3)
print("dp2:", dp2, type(dp2))
assert isinstance(dp2, dparray.ndarray)
assert dparray.has_array_interface(dp2)
del nd3
del dp2

print("------------------- Testing Numba dparray.empty")
sys.stdout.flush()
dp3 = f10()
print("dp3:", dp3, type(dp3))
assert isinstance(dp3, dparray.ndarray)
assert dparray.has_array_interface(dp3)

print("------------------- Testing Numba dparray.shape")
sys.stdout.flush()
s1 = f11(dp3)
print("s1:", s1, type(s1))

print("------------------- Testing Numba dparray.T")
sys.stdout.flush()
dp4 = f12(dp3)
print("dp4:", dp4, type(dp4))
assert isinstance(dp4, dparray.ndarray)
assert dparray.has_array_interface(dp4)
del dp3
del dp4

# -------------------------------
del a

print("SUCCESS")
