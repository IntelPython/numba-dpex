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

from numba_dppy.ocl.stubs import Stub


class dpnp(Stub):
    """dpnp namespace"""

    _description_ = "<dpnp>"

    class sum(Stub):
        pass

    class eig(Stub):
        pass

    class prod(Stub):
        pass

    class max(Stub):
        pass

    class amax(Stub):
        pass

    class min(Stub):
        pass

    class amin(Stub):
        pass

    class mean(Stub):
        pass

    class median(Stub):
        pass

    class argmax(Stub):
        pass

    class argmin(Stub):
        pass

    class argsort(Stub):
        pass

    class cov(Stub):
        pass

    class dot(Stub):
        pass

    class matmul(Stub):
        pass

    class random_sample(Stub):
        pass

    class ranf(Stub):
        pass

    class sample(Stub):
        pass

    class random(Stub):
        pass

    class rand(Stub):
        pass

    class randint(Stub):
        pass

    class random_integers(Stub):
        pass

    class beta(Stub):
        pass

    class binomial(Stub):
        pass

    class chisquare(Stub):
        pass

    class exponential(Stub):
        pass

    class gamma(Stub):
        pass

    class geometric(Stub):
        pass

    class gumbel(Stub):
        pass

    class hypergeometric(Stub):
        pass

    class laplace(Stub):
        pass

    class lognormal(Stub):
        pass

    class multinomial(Stub):
        pass

    class multivariate_normal(Stub):
        pass

    class negative_binomial(Stub):
        pass

    class normal(Stub):
        pass

    class poisson(Stub):
        pass

    class rayleigh(Stub):
        pass

    class standard_cauchy(Stub):
        pass

    class standard_exponential(Stub):
        pass

    class standard_gamma(Stub):
        pass

    class standard_normal(Stub):
        pass

    class uniform(Stub):
        pass

    class weibull(Stub):
        pass

    class vdot(Stub):
        pass

    class cholesky(Stub):
        pass

    class det(Stub):
        pass

    class multi_dot(Stub):
        pass

    class matrix_power(Stub):
        pass

    class matrix_rank(Stub):
        pass

    class eigvals(Stub):
        pass

    class nansum(Stub):
        pass

    class nanprod(Stub):
        pass

    class full(Stub):
        pass

    class ones_like(Stub):
        pass

    class zeros_like(Stub):
        pass

    class full_like(Stub):
        pass

    class copy(Stub):
        pass

    class cumsum(Stub):
        pass

    class cumprod(Stub):
        pass

    class sort(Stub):
        pass

    class take(Stub):
        pass

    class trace(Stub):
        pass
