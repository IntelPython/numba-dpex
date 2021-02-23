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
class Testdpnp_random_functions(unittest.TestCase):
    sizes = [None, 9, (2, 5), (3, 2, 4)]

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



@unittest.skipUnless(
    ensure_dpnp() and dpctl.has_gpu_queues(), "test only when dpNP and GPU is available"
)
class Testdpnp_functions(unittest.TestCase):
    N = 10

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]


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


if __name__ == "__main__":
    unittest.main()
