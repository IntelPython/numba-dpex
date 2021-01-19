#! /usr/bin/env python
from timeit import default_timer as time

import sys
import numpy as np
from numba import njit
import numba_dppy
import numba_dppy as dppy
import dpctl
import unittest
from numba_dppy.testing import ensure_dpnp, set_dpnp_debug

import dpctl


def test_for_different_datatypes(
    fn, test_fn, dims, arg_count, tys, np_all=False, matrix=None
):
    if arg_count == 1:
        for ty in tys:
            if matrix and matrix[0]:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty).reshape(
                    dims[0], dims[1]
                )
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
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty).reshape(
                    dims[0], dims[1]
                )
            else:
                a = np.array(np.random.random(dims[0] * dims[1]), dtype=ty)
            if matrix and matrix[1]:
                b = np.array(np.random.random(dims[2] * dims[3]), dtype=ty).reshape(
                    dims[2], dims[3]
                )
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


# From https://github.com/IntelPython/dpnp/blob/0.4.0/tests/test_linalg.py#L8
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


def sample_matrix(m, dtype, order="C"):
    # pd. (positive definite) matrix has eigenvalues in Z+
    np.random.seed(0)  # repeatable seed
    A = np.random.rand(m, m)
    # orthonormal q needed to form up q^{-1}*D*q
    # no "orth()" in numpy
    q, _ = np.linalg.qr(A)
    L = np.arange(1, m + 1)  # some positive eigenvalues
    Q = np.dot(np.dot(q.T, np.diag(L)), q)  # construct
    Q = np.array(Q, dtype=dtype, order=order)  # sort out order/type
    return Q


@unittest.skipUnless(ensure_dpnp(), "test only when dpNP is available")
class Testdpnp_linalg_functions(unittest.TestCase):
    #tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]
    tys = [ np.float, np.double]

    def test_eig(self):
        @njit
        def f(a):
            return np.linalg.eig(a)

        size = 3
        for ty in self.tys:
            a = np.arange(size * size, dtype=ty).reshape((size, size))
            symm_a = (
                np.tril(a)
                + np.tril(a, -1).T
                + np.diag(np.full((size,), size * size, dtype=ty))
            )

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

    def test_matmul(self):
        @njit
        def f(a, b):
            c = np.matmul(a, b)
            return c

        self.assertTrue(
            test_for_different_datatypes(
                f,
                np.matmul,
                [10, 5, 5, 10],
                2,
                [np.float, np.double],
                np_all=True,
                matrix=[True, True],
            )
        )

    def test_dot(self):
        @njit
        def f(a, b):
            c = np.dot(a, b)
            return c

        self.assertTrue(
            test_for_different_datatypes(
                f, np.dot, [10, 1, 10, 1], 2, [np.float, np.double]
            )
        )
        self.assertTrue(
            test_for_different_datatypes(
                f,
                np.dot,
                [10, 1, 10, 2],
                2,
                [np.float, np.double],
                matrix=[False, True],
                np_all=True,
            )
        )
        self.assertTrue(
            test_for_different_datatypes(
                f,
                np.dot,
                [2, 10, 10, 1],
                2,
                [np.float, np.double],
                matrix=[True, False],
                np_all=True,
            )
        )
        self.assertTrue(
            test_for_different_datatypes(
                f,
                np.dot,
                [10, 2, 2, 10],
                2,
                [np.float, np.double],
                matrix=[True, True],
                np_all=True,
            )
        )

    @unittest.skip("")
    def test_cholesky(self):
        @njit
        def f(a):
            c = np.linalg.cholesky(a)
            return c

        with dpctl.device_context("opencl:gpu"):
            for ty in self.tys:
                a = np.array([[1, -2], [2, 5]], dtype=ty)
                got = f(a)
                expected = np.linalg.cholesky(a)
                self.assertTrue(np.array_equal(got, expected))

    def test_det(self):
        @njit
        def f(a):
            c = np.linalg.det(a)
            return c

        arrays = [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]
	]

        with dpctl.device_context("opencl:gpu"):
            for ary in arrays:
                for ty in self.tys:
                    a = np.array(ary, dtype=ty)
                    got = f(a)
                    expected = np.linalg.det(a)
                    print(got, expected)
                    self.assertTrue(np.array_equal(got, expected))


@unittest.skipUnless(ensure_dpnp(), "test only when dpNP is available")
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

    def test_ndarray_prod(self):
        @njit
        def f(a):
            return a.prod()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.prod()

            self.assertTrue(expected == got)

    def test_ndarray_max(self):
        @njit
        def f(a):
            return a.max()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.max()

            self.assertTrue(expected == got)

    def test_ndarray_min(self):
        @njit
        def f(a):
            return a.min()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.min()

            self.assertTrue(expected == got)

    def test_ndarray_mean(self):
        @njit
        def f(a):
            return a.mean()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.mean()

            self.assertTrue(expected == got)

    def test_ndarray_argmax(self):
        @njit
        def f(a):
            return a.argmax()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.argmax()

            self.assertTrue(expected == got)

    def test_ndarray_argmin(self):
        @njit
        def f(a):
            return a.argmin()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty).reshape((size, size))

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.argmin()

            self.assertTrue(expected == got)

    def test_ndarray_argsort(self):
        @njit
        def f(a):
            return a.argsort()

        size = 3
        for ty in self.tys:
            a = np.arange(1, (size * size) + 1, dtype=ty)

            with dpctl.device_context("opencl:gpu"):
                got = f(a)
                expected = a.argsort()

            self.assertTrue(np.array_equal(expected, got))


@unittest.skipUnless(ensure_dpnp(), "test only when dpNP is available")
class Testdpnp_random_functions(unittest.TestCase):
    def test_random_sample(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(size):
            c = np.random.random_sample(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= 0.0)
                        self.assertTrue(_result[i] < 1.0)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())
        set_dpnp_debug(None)

    def test_ranf(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(size):
            c = np.random.ranf(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= 0.0)
                        self.assertTrue(_result[i] < 1.0)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())
        set_dpnp_debug(None)

    def test_sample(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(size):
            c = np.random.sample(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= 0.0)
                        self.assertTrue(_result[i] < 1.0)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())
        set_dpnp_debug(None)

    def test_random(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(size):
            c = np.random.random(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= 0.0)
                        self.assertTrue(_result[i] < 1.0)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())
        set_dpnp_debug(None)

    def test_rand(self):
        from numba.tests.support import captured_stdout
        @njit
        def f():
            c = np.random.rand(3, 2)
            return c

        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                result = f()
                _result = result.ravel()
                for i in range(_result.size):
                    self.assertTrue(_result[i] >= 0.0)
                    self.assertTrue(_result[i] < 1.0)
                self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())
        set_dpnp_debug(None)

    def test_randint(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(low, high, size):
            c = np.random.randint(low, high=high, size=size)
            return c

        @njit
        def f1(low, size):
            c = np.random.randint(low, size=size)
            return c

        @njit
        def f2(low, high):
            c = np.random.randint(low, high=high)
            return c

        @njit
        def f3(low):
            c = np.random.randint(low)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        low = 2
        high = 23
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(low, high, size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= low)
                        self.assertTrue(_result[i] < high)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())

                result = f(low, None, sizes[0])
                _result = result.ravel()

                for i in range(_result.size):
                    self.assertTrue(_result[i] >= 0)
                    self.assertTrue(_result[i] < low)

                result = f1(low, sizes[0])
                _result = result.ravel()

                for i in range(_result.size):
                    self.assertTrue(_result[i] >= 0)
                    self.assertTrue(_result[i] < low)

                result = f2(low, high)

                self.assertTrue(result[0] >= low)
                self.assertTrue(result[0] < high)

                result = f3(low)

                self.assertTrue(result[0] >= 0)
                self.assertTrue(result[0] < low)

        set_dpnp_debug(None)

    def test_random_integers(self):
        from numba.tests.support import captured_stdout
        @njit
        def f(low, high, size):
            c = np.random.random_integers(low, high=high, size=size)
            return c

        @njit
        def f1(low, size):
            c = np.random.random_integers(low, size=size)
            return c

        @njit
        def f2(low, high):
            c = np.random.random_integers(low, high=high)
            return c

        @njit
        def f3(low):
            c = np.random.random_integers(low)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        low = 2
        high = 23
        set_dpnp_debug(1)
        with captured_stdout() as got_gpu_message:
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(low, high, size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= low)
                        self.assertTrue(_result[i] <= high)
                    self.assertTrue("DPNP implementation" in got_gpu_message.getvalue())

                result = f(low, None, sizes[0])
                _result = result.ravel()

                for i in range(_result.size):
                    self.assertTrue(_result[i] >= 1)
                    self.assertTrue(_result[i] <= low)

                result = f1(low, sizes[0])
                _result = result.ravel()

                for i in range(_result.size):
                    self.assertTrue(_result[i] >= 1)
                    self.assertTrue(_result[i] <= low)

                result = f2(low, high)

                self.assertTrue(result[0] >= low)
                self.assertTrue(result[0] <= high)

                result = f3(low)

                self.assertTrue(result[0] >= 1)
                self.assertTrue(result[0] <= low)


        set_dpnp_debug(None)


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
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

        self.assertTrue(test_for_different_datatypes(f, np.sum, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.sum, [10, 2, 3], self.tys))

    def test_prod(self):
        @njit
        def f(a):
            c = np.prod(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.prod, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.prod, [10, 2, 3], self.tys))

    def test_argmax(self):
        @njit
        def f(a):
            c = np.argmax(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmax, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmax, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmax, [10, 2, 3], self.tys))

    def test_max(self):
        @njit
        def f(a):
            c = np.max(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.max, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.max, [10, 2, 3], self.tys))

    def test_amax(self):
        @njit
        def f(a):
            c = np.amax(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.amax, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.amax, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.amax, [10, 2, 3], self.tys))

    def test_argmin(self):
        @njit
        def f(a):
            c = np.argmin(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.argmin, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmin, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.argmin, [10, 2, 3], self.tys))

    def test_min(self):
        @njit
        def f(a):
            c = np.min(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.min, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_amin(self):
        @njit
        def f(a):
            c = np.amin(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.min, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_argsort(self):
        @njit
        def f(a):
            c = np.argsort(a)
            return c

        self.assertTrue(
            test_for_different_datatypes(f, np.argmin, [10], 1, self.tys, np_all=True)
        )

    def test_median(self):
        @njit
        def f(a):
            c = np.median(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.median, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.median, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.median, [10, 2, 3], self.tys))

    def test_mean(self):
        @njit
        def f(a):
            c = np.mean(a)
            return c

        self.assertTrue(test_for_different_datatypes(f, np.mean, [10], 1, self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2], self.tys))
        self.assertTrue(test_for_dimensions(f, np.mean, [10, 2, 3], self.tys))

    def test_cov(self):
        @njit
        def f(a):
            c = np.cov(a)
            return c

        self.assertTrue(
            test_for_different_datatypes(
                f, np.cov, [10, 7], 1, self.tys, matrix=[True], np_all=True
            )
        )

    def test_dpnp_interacting_with_parfor(self):
        @njit
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            # d = a + 1
            return 0

        result = f(self.a, self.b)
        # np_result = np.add((self.a + np.sum(self.a)), self.b)

        # max_abs_err = result.sum() - np_result.sum()
        # self.assertTrue(max_abs_err < 1e-4)


if __name__ == "__main__":
    unittest.main()
