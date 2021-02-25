#! /usr/bin/env python
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

import numpy as np
from numba import njit
import dpctl
import unittest
from numba_dppy.testing import ensure_dpnp, assert_dpnp_implementaion


def check_for_different_datatypes(
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


def check_for_different_datatypes_array_creations(
    fn, test_fn, dims, arg_count, tys, np_all=False, matrix=None, fill_value=10, func=0
):
    for ty in tys:
        a = np.array(np.random.random(dims[0]), dtype=ty)
        b_dppy = np.array([fill_value], dtype=ty)
        b_numpy = b_dppy[0]

        if arg_count == 1:
            with dpctl.device_context("opencl:gpu"):
                c = fn(a)
            d = test_fn(a)
        elif arg_count == 2:
            if func == 2:
                with dpctl.device_context("opencl:gpu"):
                    c = fn(a, dtype=ty)
                d = test_fn(a, dtype=ty)
            elif func == 1:
                with dpctl.device_context("opencl:gpu"):
                    c = fn(a, b_dppy)
                d = test_fn(a, b_numpy)
            else:
                with dpctl.device_context("opencl:gpu"):
                    c = fn(a, b_dppy)
                d = test_fn(a.shape, b_numpy)

        if np_all:
            max_abs_err = np.all(c - d)
        else:
            max_abs_err = c - d
        if not (max_abs_err < 1e-4):
            return False
    return True


def check_for_dimensions(fn, test_fn, dims, tys, np_all=False):
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

        if not (vec is None):
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
    tys = [np.int32, np.int64, np.float, np.double]

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
            check_for_different_datatypes(
                f,
                np.matmul,
                [10, 5, 5, 10],
                2,
                [np.int32, np.int64, np.float, np.double],
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
            check_for_different_datatypes(
                f, np.dot, [10, 1, 10, 1], 2, [np.int32, np.int64, np.float, np.double]
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.dot,
                [10, 1, 10, 2],
                2,
                [np.int32, np.int64, np.float, np.double],
                matrix=[False, True],
                np_all=True,
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.dot,
                [2, 10, 10, 1],
                2,
                [np.int32, np.int64, np.float, np.double],
                matrix=[True, False],
                np_all=True,
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.dot,
                [10, 2, 2, 10],
                2,
                [np.int32, np.int64, np.float, np.double],
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

    @unittest.skip("")
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
            [
                [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
                [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
            ],
        ]

        with dpctl.device_context("opencl:gpu"):
            for ary in arrays:
                for ty in self.tys:
                    a = np.array(ary, dtype=ty)
                    got = f(a)
                    expected = np.linalg.det(a)
                    self.assertTrue(np.array_equal(got, expected))

    def test_multi_dot(self):
        @njit
        def f(A, B, C, D):
            c = np.linalg.multi_dot([A, B, C, D])
            return c

        A = np.random.random((10000, 100))
        B = np.random.random((100, 1000))
        C = np.random.random((1000, 5))
        D = np.random.random((5, 333))

        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                got = f(A, B, C, D)

        expected = np.linalg.multi_dot([A, B, C, D])
        self.assertTrue(np.allclose(got, expected, atol=1e-04))

    def test_vdot(self):
        @njit
        def f(a, b):
            c = np.vdot(a, b)
            return c

        self.assertTrue(
            check_for_different_datatypes(
                f, np.vdot, [10, 1, 10, 1], 2, [np.int32, np.int64, np.float, np.double]
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.vdot,
                [10, 1, 10, 1],
                2,
                [np.int32, np.int64, np.float, np.double],
                matrix=[False, True],
                np_all=True,
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.vdot,
                [2, 10, 10, 2],
                2,
                [np.int32, np.int64, np.float, np.double],
                matrix=[True, False],
                np_all=True,
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
                f,
                np.vdot,
                [10, 2, 2, 10],
                2,
                [np.int32, np.int64, np.float, np.double],
                matrix=[True, True],
                np_all=True,
            )
        )

    def test_matrix_power(self):
        @njit
        def f(a, n):
            c = np.linalg.matrix_power(a, n)
            return c

        arrays = [
            [[0, 0], [0, 0]],
            [[1, 2], [1, 2]],
            [[1, 2], [3, 4]],
        ]

        ns = [2, 3, 0]
        with dpctl.device_context("opencl:gpu"):
            for n in ns:
                for ary in arrays:
                    for ty in self.tys:
                        a = np.array(ary, dtype=ty)
                        got = f(a, n)
                        expected = np.linalg.matrix_power(a, n)
                        self.assertTrue(np.allclose(got, expected))

    @unittest.skip("")
    def test_matrix_rank(self):
        @njit
        def f(a):
            c = np.linalg.matrix_rank(a)
            return c

        arrays = [np.eye(4), np.ones((4,)), np.ones((4, 4)), np.zeros((4,))]

        with dpctl.device_context("opencl:gpu"):
            for ary in arrays:
                for ty in self.tys:
                    a = np.array(ary, dtype=ty)
                    got = f(a)
                    expected = np.linalg.matrix_rank(a)
                    print(got, expected)
                    self.assertTrue(np.allclose(got, expected))

    def test_eigvals(self):
        @njit
        def f(a):
            return np.linalg.eigvals(a)

        size = 3
        for ty in self.tys:
            a = np.arange(size * size, dtype=ty).reshape((size, size))
            symm_a = (
                np.tril(a)
                + np.tril(a, -1).T
                + np.diag(np.full((size,), size * size, dtype=ty))
            )

            with dpctl.device_context("opencl:gpu"):
                got_val = f(symm_a)

            np_val = np.linalg.eigvals(symm_a)

            # sort val by abs value
            vvsort(got_val, None, size)
            vvsort(np_val, None, size)

            self.assertTrue(np.allclose(got_val, np_val))


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
    sizes = [None, 9, (2, 5), (3, 2, 4)]

    def test_random_sample(self):
        @njit
        def f(size):
            c = np.random.random_sample(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0.0)
                self.assertTrue(_result[i] < 1.0)

    def test_ranf(self):
        @njit
        def f(size):
            c = np.random.ranf(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0.0)
                self.assertTrue(_result[i] < 1.0)

    def test_sample(self):
        @njit
        def f(size):
            c = np.random.sample(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0.0)
                self.assertTrue(_result[i] < 1.0)

    def test_random(self):
        @njit
        def f(size):
            c = np.random.random(size)
            return c

        sizes = [9, (2, 5), (3, 2, 4)]
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0.0)
                self.assertTrue(_result[i] < 1.0)

    def test_rand(self):
        @njit
        def f():
            c = np.random.rand(3, 2)
            return c

        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                result = f()

        _result = result.ravel()
        for i in range(_result.size):
            self.assertTrue(_result[i] >= 0.0)
            self.assertTrue(_result[i] < 1.0)

    def test_randint(self):
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
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(low, high, size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= low)
                self.assertTrue(_result[i] < high)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(low, None, sizes[0])
            _result = result.ravel()

            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0)
                self.assertTrue(_result[i] < low)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f1(low, sizes[0])

            _result = result.ravel()

            for i in range(_result.size):
                self.assertTrue(_result[i] >= 0)
                self.assertTrue(_result[i] < low)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f2(low, high)

            self.assertTrue(result[0] >= low)
            self.assertTrue(result[0] < high)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f3(low)

            self.assertTrue(result[0] >= 0)
            self.assertTrue(result[0] < low)

    def test_random_integers(self):
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
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(low, high, size)

            _result = result.ravel()
            for i in range(_result.size):
                self.assertTrue(_result[i] >= low)
                self.assertTrue(_result[i] <= high)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(low, None, sizes[0])

            _result = result.ravel()

            for i in range(_result.size):
                self.assertTrue(_result[i] >= 1)
                self.assertTrue(_result[i] <= low)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f1(low, sizes[0])

            _result = result.ravel()

            for i in range(_result.size):
                self.assertTrue(_result[i] >= 1)
                self.assertTrue(_result[i] <= low)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f2(low, high)

            self.assertTrue(result[0] >= low)
            self.assertTrue(result[0] <= high)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f3(low)

            self.assertTrue(result[0] >= 1)
            self.assertTrue(result[0] <= low)

    def test_beta(self):
        @njit
        def f(a, b, size):
            res = np.random.beta(a, b, size)
            return res

        alpha = 2.56
        beta = 0.8
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(alpha, beta, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
                self.assertTrue(result <= 1.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))
                self.assertTrue(np.all(final_result <= 1.0))

    def test_binomial(self):
        @njit
        def f(a, b, size):
            res = np.random.binomial(a, b, size)
            return res

        n = 5
        p = 0.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(n, p, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))
                self.assertTrue(np.all(final_result <= n))

    def test_chisquare(self):
        @njit
        def f(df, size):
            res = np.random.chisquare(df, size)
            return res

        df = 3  # number of degrees of freedom
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(df, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    def test_exponential(self):
        @njit
        def f(scale, size):
            res = np.random.exponential(scale, size)
            return res

        scale = 3.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(scale, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    @unittest.skip("AttributeError: 'NoneType' object has no attribute 'ravel'")
    def test_gamma(self):
        @njit
        def f(shape, size):
            res = np.random.gamma(shape=shape, size=size)
            return res

        shape = 2.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(shape, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    def test_geometric(self):
        @njit
        def f(p, size):
            res = np.random.geometric(p, size=size)
            return res

        p = 0.35
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(p, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    def test_gumbel(self):
        @njit
        def f(loc, scale, size):
            res = np.random.gumbel(loc=loc, scale=scale, size=size)
            return res

        mu, beta = 0.5, 0.1  # location and scale
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(mu, beta, size)
                    # TODO: check result, x belongs R

    def test_hypergeometric(self):
        @njit
        def f(ngood, nbad, nsamp, size):
            res = np.random.hypergeometric(ngood, nbad, nsamp, size)
            return res

        ngood, nbad, nsamp = 100, 2, 10
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(ngood, nbad, nsamp, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
                self.assertTrue(result <= min(nsamp, ngood + nbad))
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))
                self.assertTrue(np.all(final_result <= min(nsamp, ngood + nbad)))

    def test_laplace(self):
        @njit
        def f(loc, scale, size):
            res = np.random.laplace(loc, scale, size)
            return res

        loc, scale = 0.0, 1.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(loc, scale, size)
                    # TODO: check result, x belongs R

    def test_lognormal(self):
        @njit
        def f(mean, sigma, size):
            res = np.random.lognormal(mean, sigma, size)
            return res

        mu, sigma = 3.0, 1.0  # mean and standard deviation
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(mu, sigma, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    @unittest.skip("DPNP RNG Error: dpnp_rng_multinomial_c() failed")
    def test_multinomial(self):
        @njit
        def f(n, pvals, size):
            res = np.random.multinomial(n, pvals, size)
            return res

        n, pvals = 100, np.array([1 / 7.0] * 5)
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(n, pvals, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
                self.assertTrue(result <= n)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))
                self.assertTrue(np.all(final_result <= n))

    @unittest.skip(
        "No implementation of function Function(<class "
        "'numba_dppy.dpnp_glue.stubs.dpnp.multivariate_normal'>) found for signature"
    )
    def test_multivariate_normal(self):
        @njit
        def f(mean, cov, size):
            res = np.random.multivariate_normal(mean, cov, size)
            return res

        mean, cov = (1, 2), [[1, 0], [0, 1]]
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(mean, cov, size)
                    # TODO: check result, for multidimensional distribution

    @unittest.skip("DPNP RNG Error: dpnp_rng_negative_binomial_c() failed.")
    def test_negative_binomial(self):
        @njit
        def f(n, p, size):
            res = np.random.negative_binomial(n, p, size)
            return res

        n, p = 1, 0.1
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(n, p, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0))

    def test_normal(self):
        @njit
        def f(loc, scale, size):
            res = np.random.normal(loc, scale, size)
            return res

        sizes = (1000,)
        mu, sigma = 0.0, 0.1  # mean and standard deviation
        for size in sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(mu, sigma, size)

            if np.isscalar(result):
                self.assertTrue(abs(mu - np.mean(result)) < 0.01)
            else:
                final_result = result.ravel()
                self.assertTrue(abs(mu - np.mean(final_result)) < 0.01)

    def test_poisson(self):
        @njit
        def f(lam, size):
            res = np.random.poisson(lam, size)
            return res

        lam = 5.0  # lambda
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(lam, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0.0))

    def test_rayleigh(self):
        @njit
        def f(scale, size):
            res = np.random.rayleigh(scale, size)
            return res

        scale = 2.0  # lambda
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(scale, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0.0))

    def test_standard_cauchy(self):
        @njit
        def f(size):
            res = np.random.standard_cauchy(size)
            return res

        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)
                    # TODO: check result, x belongs R

    def test_standard_exponential(self):
        @njit
        def f(size):
            res = np.random.standard_exponential(size)
            return res

        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)

            if np.isscalar(result):
                self.assertTrue(result >= 0.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0.0))

    def test_standard_gamma(self):
        @njit
        def f(shape, size):
            res = np.random.standard_gamma(shape, size)
            return res

        shape = 2.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(shape, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0.0))

    def test_standard_normal(self):
        @njit
        def f(size):
            res = np.random.standard_normal(size)
            return res

        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(size)
                    # TODO: check result, x belongs R

    def test_uniform(self):
        @njit
        def f(low, high, size):
            res = np.random.uniform(low=low, high=high, size=size)
            return res

        low, high = -1.0, 0.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(low, high, size)

            if np.isscalar(result):
                self.assertTrue(result >= low)
                self.assertTrue(result < high)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= low))
                self.assertTrue(np.all(final_result < high))

    def test_weibull(self):
        @njit
        def f(a, size):
            res = np.random.weibull(a, size)
            return res

        a = 5.0
        for size in self.sizes:
            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    result = f(a, size)

            if np.isscalar(result):
                self.assertTrue(result >= 0.0)
            else:
                final_result = result.ravel()
                self.assertTrue(np.all(final_result >= 0.0))


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP is available"
)
class Testdpnp_transcendentals_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]
    nantys = [np.float, np.double]

    def test_sum(self):
        @njit
        def f(a):
            c = np.sum(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(check_for_different_datatypes(f, np.sum, [10], 1, self.tys))
            self.assertTrue(check_for_dimensions(f, np.sum, [10, 2], self.tys))
            self.assertTrue(check_for_dimensions(f, np.sum, [10, 2, 3], self.tys))

    def test_prod(self):
        @njit
        def f(a):
            c = np.prod(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(f, np.prod, [10], 1, self.tys)
            )
            self.assertTrue(check_for_dimensions(f, np.prod, [10, 2], self.tys))
            self.assertTrue(check_for_dimensions(f, np.prod, [10, 2, 3], self.tys))

    def test_nansum(self):
        @njit
        def f(a):
            c = np.nansum(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(f, np.nansum, [10], 1, self.tys)
            )
            self.assertTrue(check_for_dimensions(f, np.nansum, [10, 2], self.tys))
            self.assertTrue(check_for_dimensions(f, np.nansum, [10, 2, 3], self.tys))

        a = np.array([[1, 2], [1, np.nan]])

        for ty in self.nantys:
            ary = np.array(a, dtype=ty)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    got = f(ary)

            expected = np.nansum(ary)
            max_abs_err = np.sum(got) - np.sum(expected)
            self.assertTrue(max_abs_err < 1e-4)

    def test_nanprod(self):
        @njit
        def f(a):
            c = np.nanprod(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(f, np.nanprod, [10], 1, self.tys)
            )
            self.assertTrue(check_for_dimensions(f, np.nanprod, [10, 2], self.tys))
            self.assertTrue(check_for_dimensions(f, np.nanprod, [10, 2, 3], self.tys))

        a = np.array([[1, 2], [1, np.nan]])

        for ty in self.nantys:
            ary = np.array(a, dtype=ty)

            with assert_dpnp_implementaion():
                with dpctl.device_context("opencl:gpu"):
                    got = f(ary)

            expected = np.nanprod(ary)
            max_abs_err = np.sum(got) - np.sum(expected)
            self.assertTrue(max_abs_err < 1e-4)


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
class Testdpnp_functions(unittest.TestCase):
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

    def test_argmax(self):
        @njit
        def f(a):
            c = np.argmax(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.argmax, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.argmax, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.argmax, [10, 2, 3], self.tys))

    def test_max(self):
        @njit
        def f(a):
            c = np.max(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.max, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.max, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.max, [10, 2, 3], self.tys))

    def test_amax(self):
        @njit
        def f(a):
            c = np.amax(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.amax, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.amax, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.amax, [10, 2, 3], self.tys))

    def test_argmin(self):
        @njit
        def f(a):
            c = np.argmin(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.argmin, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.argmin, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.argmin, [10, 2, 3], self.tys))

    def test_min(self):
        @njit
        def f(a):
            c = np.min(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.min, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_amin(self):
        @njit
        def f(a):
            c = np.amin(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.min, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.min, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.min, [10, 2, 3], self.tys))

    def test_argsort(self):
        @njit
        def f(a):
            c = np.argsort(a)
            return c

        self.assertTrue(
            check_for_different_datatypes(f, np.argsort, [10], 1, self.tys, np_all=True)
        )

    def test_median(self):
        @njit
        def f(a):
            c = np.median(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.median, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.median, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.median, [10, 2, 3], self.tys))

    def test_mean(self):
        @njit
        def f(a):
            c = np.mean(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.mean, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.mean, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.mean, [10, 2, 3], self.tys))

    def test_matmul(self):
        @njit
        def f(a, b):
            c = np.matmul(a, b)
            return c

        self.assertTrue(
            check_for_different_datatypes(
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
            check_for_different_datatypes(
                f, np.dot, [10, 1, 10, 1], 2, [np.float, np.double]
            )
        )
        self.assertTrue(
            check_for_different_datatypes(
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
            check_for_different_datatypes(
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
            check_for_different_datatypes(
                f,
                np.dot,
                [10, 2, 2, 10],
                2,
                [np.float, np.double],
                matrix=[True, True],
                np_all=True,
            )
        )

    def test_cov(self):
        @njit
        def f(a):
            c = np.cov(a)
            return c

        self.assertTrue(
            check_for_different_datatypes(
                f, np.cov, [10, 7], 1, self.tys, matrix=[True], np_all=True
            )
        )

    def test_dpnp_interacting_with_parfor(self):
        def f(a, b):
            c = np.sum(a)
            e = np.add(b, a)
            d = c + e
            return d

        njit_f = njit(f)
        got = njit_f(self.a, self.b)
        expected = f(self.a, self.b)

        max_abs_err = got.sum() - expected.sum()
        self.assertTrue(max_abs_err < 1e-4)


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
class Testdpnp_array_ops_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

    def test_copy(self):
        @njit
        def f(a):
            c = np.copy(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(
                    f, np.copy, [10], 1, self.tys, np_all=True
                )
            )

    def test_cumsum(self):
        @njit
        def f(a):
            c = np.cumsum(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(f, np.cumsum, [10], 1, self.tys, True)
            )
            self.assertTrue(check_for_dimensions(f, np.cumsum, [10, 2], self.tys, True))
            self.assertTrue(
                check_for_dimensions(f, np.cumsum, [10, 2, 3], self.tys, True)
            )

    def test_cumprod(self):
        @njit
        def f(a):
            c = np.cumprod(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(f, np.cumprod, [10], 1, self.tys, True)
            )
            self.assertTrue(
                check_for_dimensions(f, np.cumprod, [10, 2], self.tys, True)
            )
            self.assertTrue(
                check_for_dimensions(f, np.cumprod, [10, 2, 3], self.tys, True)
            )

    def test_sort(self):
        @njit
        def f(a):
            c = np.sort(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes(
                    f, np.sort, [10], 1, self.tys, np_all=True
                )
            )

    def check_take_for_different_datatypes(
        self, fn, test_fn, ind, dims, tys, matrix=False
    ):
        for ty in tys:
            if matrix:
                a = np.arange(np.prod(dims), dtype=ty).reshape(dims[0], dims[1])
            else:
                a = np.arange(dims[0], dtype=ty)

            c = fn(a, ind)

            d = test_fn(a, ind)
            if c.shape == d.shape:
                max_abs_err = np.all(c - d)
            if not (max_abs_err < 1e-4) and c.dtype != d.dtype:
                return False

        return True

    def test_take(self):
        @njit
        def f(a, ind):
            c = np.take(a, ind)
            return c

        test_indices = []
        test_indices.append(np.array([[1, 5, 1], [11, 3, 0]]))
        test_indices.append(np.array([[[1, 5, 1], [11, 3, 0]]]))
        test_indices.append(np.array([[[[1, 5]], [[11, 0]], [[1, 2]]]]))

        self.assertTrue(
            self.check_take_for_different_datatypes(
                f, np.take, np.array([1, 5, 1, 11, 3]), [12], self.tys
            )
        )

        for ind in test_indices:
            self.assertTrue(
                self.check_take_for_different_datatypes(
                    f,
                    np.take,
                    ind,
                    [3, 4],
                    [np.float],
                    matrix=True,
                )
            )


@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
class Testdpnp_array_creations_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float64, np.double]

    def test_full(self):
        @njit
        def f(a, b):
            c = np.full(a, b)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.full, [10], 2, self.tys, np_all=True
                )
            )

    def test_ones_like(self):
        @njit
        def f(a, dtype):
            c = np.ones_like(a, dtype)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.ones_like, [10], 2, self.tys, np_all=True, func=2
                )
            )

    def test_ones_like_without_dtype(self):
        @njit
        def f(a):
            c = np.ones_like(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.ones_like, [10], 1, self.tys, np_all=True
                )
            )

    def test_zeros_like(self):
        @njit
        def f(a, dtype):
            c = np.zeros_like(a, dtype)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.zeros_like, [10], 2, self.tys, np_all=True, func=2
                )
            )

    def test_zeros_like_without_dtype(self):
        @njit
        def f(a):
            c = np.zeros_like(a)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.zeros_like, [10], 1, self.tys, np_all=True
                )
            )

    def test_full_like(self):
        @njit
        def f(a, b):
            c = np.full_like(a, b)
            return c

        with assert_dpnp_implementaion():
            self.assertTrue(
                check_for_different_datatypes_array_creations(
                    f, np.full_like, [10], 2, self.tys, np_all=True, func=1
                )
            )


if __name__ == "__main__":
    unittest.main()
