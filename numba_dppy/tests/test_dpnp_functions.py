#! /usr/bin/env python
from timeit import default_timer as time

import sys
import numpy as np
from numba import njit
import numba_dppy
import numba_dppy as dppy
import dpctl
import unittest


import dpctl

def test_for_different_datatypes(fn, test_fn, dims, arg_count, tys, np_all=False, matrix=None):
    if arg_count == 1:
        for ty in tys:
            if matrix and matrix[0]:
                a = np.array(np.random.random(
                    dims[0] * dims[1]), dtype=ty).reshape(dims[0], dims[1])
            else:
                a = np.array(np.random.random(dims[0]), dtype=ty)

            with dpctl.device_context("opencl:gpu"):
                c = fn(a)

            d = test_fn(a)
            if np_all:
                max_abs_err = np.all(c - d)
            else:
                max_abs_err = c - d
            if not (max_abs_err < 1e-4):
                return False

    elif arg_count == 2:
        for ty in tys:
            if matrix and matrix[0]:
                a = np.array(np.random.random(
                    dims[0] * dims[1]), dtype=ty).reshape(dims[0], dims[1])
            else:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty)
            if matrix and matrix[1]:
                b = np.array(np.random.random(
                    dims[2] * dims[3]), dtype=ty).reshape(dims[2], dims[3])
            else:
                b = np.array(np.random.random(dims[2] * dims[3]), dtype=ty)

            with dpctl.device_context("opencl:gpu"):
                c = fn(a, b)

            d = test_fn(a, b)
            if np_all:
                max_abs_err = np.sum(c - d)
            else:
                max_abs_err = c - d
            if not (max_abs_err < 1e-4):
                return False

    return True


def test_for_dimensions(fn, test_fn, dims, tys, np_all=False):
    total_size = 1
    for d in dims:
        total_size *= d

    for ty in tys:
        a = np.array(np.random.random(total_size), dtype=ty).reshape(dims)

        with dpctl.device_context("opencl:gpu"):
            c = fn(a)

        d = test_fn(a)
        if np_all:
            max_abs_err = np.all(c - d)
        else:
            max_abs_err = c - d
        if not (max_abs_err < 1e-4):
            return False

    return True


def ensure_dpnp():
    try:
        from numba_dppy.dpnp_glue import dpnp_fptr_interface as dpnp_glue
        return True
    except:
        return False

# From https://github.com/IntelPython/dpnp/blob/master/tests/test_linalg.py
def vvsort(val, vec, size):
    for i in range(size):
        imax = i
        for j in range(i + 1, size):
            if np.abs(val[imax]) < np.abs(val[j]):
                imax = j

        temp = val[i]
        val[i] = val[imax]
        val[imax] = temp

        for k in range(size):
            temp = vec[k, i]
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@unittest.skipUnless(ensure_dpnp(), 'test only when dpNP is available')
class Testdpnp_linalg_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]
    def test_eig(self):
        @njit
        def f(a):
            return np.linalg.eig(a)

        size = 3
        for ty in self.tys:
            a = np.arange(size * size, dtype=ty).reshape((size, size))
            symm_a = np.tril(a) + np.tril(a, -1).T + np.diag(np.full((size,), size * size, dtype=ty))

            with dpctl.device_context("opencl:gpu"):
                got_val, got_vec = f(symm_a)

            np_val, np_vec = np.linalg.eig(symm_a)

            # sort val/vec by abs value
            vvsort(got_val, got_vec, size)
            vvsort(np_val, np_vec, size)


	    # NP change sign of vectors
            for i in range(np_vec.shape[1]):
                if np_vec[0, i] * got_vec[0, i] < 0:
                    np_vec[:, i] = -np_vec[:, i]

            self.assertTrue(np.allclose(got_val, np_val))
            self.assertTrue(np.allclose(got_vec, np_vec))


@unittest.skipUnless(ensure_dpnp(), 'test only when dpNP is available')
class Testdpnp_ndarray_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]
    def test_ndarray_sum(self):
        @njit
        def f(a):
            return a.sum()

        size = 3
        for ty in self.tys:
            a = np.arange(size * size, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.sum()

            self.assertTrue(expected == got)

@unittest.skipUnless(ensure_dpnp() and dpctl.has_gpu_queues(), 'test only when dpNP and GPU is available')
class Testdpnp_functions(unittest.TestCase):

    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

    def test_sum(self):
        @njit
        def f(a):
            c = np.sum(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.sum, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2, 3], self.tys))

    def test_prod(self):
        @njit
        def f(a):
            c = np.prod(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.prod, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2, 3], self.tys))

    def test_argmax(self):
        @njit
        def f(a):
            c = np.argmax(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.argmax, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmax, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(
            f, np.argmax, [10, 2, 3], self.tys))

    def test_max(self):
        @njit
        def f(a):
            c = np.max(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.max, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2, 3], self.tys))

    def test_argmin(self):
        @njit
        def f(a):
            c = np.argmin(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.argmin, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmin, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(
            f, np.argmin, [10, 2, 3], self.tys))

    def test_min(self):
        @njit
        def f(a):
            c = np.min(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.min, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_argsort(self):
        @njit
        def f(a):
            c = np.argsort(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.argmin, [10], 1, self.tys, np_all=True))

    def test_median(self):
        @njit
        def f(a):
            c = np.median(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.median, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.median, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(
            f, np.median, [10, 2, 3], self.tys))

    def test_mean(self):
        @njit
        def f(a):
            c = np.mean(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.mean, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2, 3], self.tys))

    def test_matmul(self):
        @njit
        def f(a, b):
            c = np.matmul(a, b)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.matmul, [10, 5, 5, 10], 2, [
                        np.float, np.double], np_all=True, matrix=[True, True]))

    def test_dot(self):
        @njit
        def f(a, b):
            c = np.dot(a, b)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.dot, [10, 1, 10, 1], 2, [np.float, np.double]))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [10, 1, 10, 2], 2, [
                        np.float, np.double], matrix=[False, True], np_all=True))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [2, 10, 10, 1], 2, [
                        np.float, np.double], matrix=[True, False], np_all=True))
        self.assertTrue(test_for_different_datatypes(f, np.dot, [10, 2, 2, 10], 2, [
                        np.float, np.double], matrix=[True, True], np_all=True))

    def test_cov(self):
        @njit
        def f(a):
            c = np.cov(a)
            return c

        self.assertTrue(test_for_different_datatypes(
            f, np.cov, [10, 7], 1, self.tys, matrix=[True], np_all=True))

    def test_dpnp_interacting_with_parfor(self):
        @njit
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            #d = a + 1
            return 0

        result = f(self.a, self.b)
        #np_result = np.add((self.a + np.sum(self.a)), self.b)

        #max_abs_err = result.sum() - np_result.sum()
        #self.assertTrue(max_abs_err < 1e-4)


if __name__ == '__main__':
    unittest.main()
