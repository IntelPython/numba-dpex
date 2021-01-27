#! /usr/bin/env python
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

        for k in range(size):
            temp = vec[k, i]
            vec[k, i] = vec[k, imax]
            vec[k, imax] = temp


@unittest.skipUnless(ensure_dpnp(), "test only when dpNP is available")
class Testdpnp_linalg_functions(unittest.TestCase):
    tys = [np.int32, np.uint32, np.int64, np.uint64, np.float, np.double]

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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(low, high, size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= low)
                        self.assertTrue(_result[i] < high)

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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
                    result = f(low, high, size)
                    _result = result.ravel()
                    for i in range(_result.size):
                        self.assertTrue(_result[i] >= low)
                        self.assertTrue(_result[i] <= high)

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

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_beta(self):
        @njit
        def f(a, b, size):
            res = np.random.beta(a, b, size)
            return res

        alpha = 2.56
        beta = 0.8
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(alpha, beta, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                        self.assertTrue(result <= 1.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)
                        self.assertTrue(final_result.all() <= 1.0)

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_binomial(self):
        @njit
        def f(a, b, size):
            res = np.random.binomial(a, b, size)
            return res

        n = 5
        p = 0.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(n, p, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)
                        self.assertTrue(final_result.all() <= n)

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_chisquare(self):
        @njit
        def f(df, size):
            res = np.random.chisquare(df, size)
            return res

        df = 3  # number of degrees of freedom
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(df, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    def test_exponential(self):
        @njit
        def f(scale, size):
            res = np.random.exponential(scale, size)
            return res

        scale = 3.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(scale, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_gamma(self):
        @njit
        def f(shape, size):
            res = np.random.gamma(shape=shape, size=size)
            return res

        shape = 2.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(shape, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    def test_geometric(self):
        @njit
        def f(p, size):
            res = np.random.geometric(p, size=size)
            return res

        p = 0.35
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(p, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    def test_gumbel(self):
        @njit
        def f(loc, scale, size):
            res = np.random.gumbel(loc=loc, scale=scale, size=size)
            return res

        mu, beta = 0.5, 0.1  # location and scale
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(mu, beta, size)
                    # TODO: check result, x belongs R

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_hypergeometric(self):
        @njit
        def f(ngood, nbad, nsamp, size):
            res = np.random.hypergeometric(ngood, nbad, nsamp, size)
            return res

        ngood, nbad, nsamp = 100, 2, 10
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(ngood, nbad, nsamp, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                        self.assertTrue(result <= min(nsamp, ngood + nbad))
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)
                        self.assertTrue(final_result.all() <= min(nsamp, ngood + nbad))

    def test_laplace(self):
        @njit
        def f(loc, scale, size):
            res = np.random.laplace(loc, scale, size)
            return res

        loc, scale = 0.0, 1.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(loc, scale, size)
                    # TODO: check result, x belongs R

    def test_lognormal(self):
        @njit
        def f(mean, sigma, size):
            res = np.random.lognormal(mean, sigma, size)
            return res

        mu, sigma = 3.0, 1.0  # mean and standard deviation
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(mu, sigma, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_multinomial(self):
        @njit
        def f(n, pvals, size):
            res = np.random.multinomial(n, pvals, size)
            return res

        n, pvals = 100, np.array([1 / 7.0] * 5)
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(n, pvals, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                        self.assertTrue(result <= n)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)
                        self.assertTrue(final_result.all() <= n)

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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(mean, cov, size)
                    # TODO: check result, for multidimensional distribution

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_negative_binomial(self):
        @njit
        def f(n, p, size):
            res = np.random.negative_binomial(n, p, size)
            return res

        n, p = 1, 0.1
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(n, p, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0)

    def test_normal(self):
        @njit
        def f(loc, scale, size):
            res = np.random.normal(loc, scale, size)
            return res

        sizes = (1000,)
        mu, sigma = 0.0, 0.1  # mean and standard deviation
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in sizes:
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
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(lam, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)

    def test_rayleigh(self):
        @njit
        def f(scale, size):
            res = np.random.rayleigh(scale, size)
            return res

        scale = 2.0  # lambda
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(scale, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)

    def test_standard_cauchy(self):
        @njit
        def f(size):
            res = np.random.standard_cauchy(size)
            return res

        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(size)
                    # TODO: check result, x belongs R

    def test_standard_exponential(self):
        @njit
        def f(size):
            res = np.random.standard_exponential(size)
            return res

        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)

    @unittest.skip("Exception from MKL, oneMKL: rng/generate")
    def test_standard_gamma(self):
        @njit
        def f(shape, size):
            res = np.random.standard_gamma(shape, size)
            return res

        shape = 2.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(shape, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)

    def test_standard_normal(self):
        @njit
        def f(size):
            res = np.random.standard_normal(size)
            return res

        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)

    @unittest.skip(
        "TypeError: dpnp_random_impl() got an unexpected keyword argument 'low'"
    )
    def test_uniform(self):
        @njit
        def f(low, high, size):
            res = np.random.standard_normal(low=low, high=high, size=size)
            return res

        low, high = -1.0, 0.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(low, high, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= low)
                        self.assertTrue(result < high)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= low)
                        self.assertTrue(final_result.all() < high)

    def test_weibull(self):
        @njit
        def f(a, size):
            res = np.random.weibull(a, size)
            return res

        a = 5.0
        with assert_dpnp_implementaion():
            with dpctl.device_context("opencl:gpu"):
                for size in self.sizes:
                    result = f(a, size)
                    if np.isscalar(result):
                        self.assertTrue(result >= 0.0)
                    else:
                        final_result = result.ravel()
                        self.assertTrue(final_result.all() >= 0.0)


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

        self.assertTrue(check_for_different_datatypes(f, np.sum, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.sum, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.sum, [10, 2, 3], self.tys))

    def test_prod(self):
        @njit
        def f(a):
            c = np.prod(a)
            return c

        self.assertTrue(check_for_different_datatypes(f, np.prod, [10], 1, self.tys))
        self.assertTrue(check_for_dimensions(f, np.prod, [10, 2], self.tys))
        self.assertTrue(check_for_dimensions(f, np.prod, [10, 2, 3], self.tys))

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
            check_for_different_datatypes(f, np.argmin, [10], 1, self.tys, np_all=True)
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
