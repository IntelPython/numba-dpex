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

rewrite_function_name_map = {
    "prod": (["numpy"], "prod"),
    "sum": (["numpy"], "sum"),
    "nanprod": (["numpy"], "nanprod"),
    "nansum": (["numpy"], "nansum"),
    "median": (["numpy"], "median"),
    "cov": (["numpy"], "cov"),
    "argsort": (["numpy"], "argsort"),
    "argmax": (["numpy"], "argmax"),
    "argmin": (["numpy"], "argmin"),
    "beta": {"random": (["numpy"], "beta")},
    "binomial": {"random": (["numpy"], "binomial")},
    "chisquare": {"random": (["numpy"], "chisquare")},
    "exponential": {"random": (["numpy"], "exponential")},
    "gamma": {"random": (["numpy"], "gamma")},
    "geometric": {"random": (["numpy"], "geometric")},
    "gumbel": {"random": (["numpy"], "gumbel")},
    "hypergeometric": {"random": (["numpy"], "hypergeometric")},
    "laplace": {"random": (["numpy"], "laplace")},
    "lognormal": {"random": (["numpy"], "lognormal")},
    "multinomial": {"random": (["numpy"], "multinomial")},
    "multivariate_normal": {"random": (["numpy"], "multivariate_normal")},
    "negative_binomial": {"random": (["numpy"], "negative_binomial")},
    "normal": {"random": (["numpy"], "normal")},
    "poisson": {"random": (["numpy"], "poisson")},
    "rand": {"random": (["numpy"], "rand")},
    "randint": {"random": (["numpy"], "randint")},
    "random_integers": {"random": (["numpy"], "random_integers")},
    "random_sample": {"random": (["numpy"], "random_sample")},
    "random": {"random": (["numpy"], "random")},
    "ranf": {"random": (["numpy"], "ranf")},
    "rayleigh": {"random": (["numpy"], "rayleigh")},
    "sample": {"random": (["numpy"], "sample")},
    "standard_cauchy": {"random": (["numpy"], "standard_cauchy")},
    "standard_exponential": {"random": (["numpy"], "standard_exponential")},
    "standard_gamma": {"random": (["numpy"], "standard_gamma")},
    "standard_normal": {"random": (["numpy"], "standard_normal")},
    "uniform": {"random": (["numpy"], "uniform")},
    "weibull": {"random": (["numpy"], "weibull")},
    "vdot": (["numpy"], "vdot"),
    "matmul": (["numpy"], "matmul"),
    "matrix_power": {"linalg": (["numpy"], "matrix_power")},
    "cholesky": {"linalg": (["numpy"], "cholesky")},
    "eig": {"linalg": (["numpy"], "eig")},
    "eigvals": {"linalg": (["numpy"], "eigvals")},
    "det": {"linalg": (["numpy"], "det")},
    "matrix_rank": {"linalg": (["numpy"], "matrix_rank")},
    "multi_dot": {"linalg": (["numpy"], "multi_dot")},
    "amax": (["numpy"], "amax"),
    "amin": (["numpy"], "amin"),
    "dot": (["numpy"], "dot"),
    "max": (["numpy"], "max"),
    "mean": (["numpy"], "mean"),
    "min": (["numpy"], "min"),
}
numba_dppy_numpy_ufunc = [
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("asin", np.arcsin),
    ("acos", np.arccos),
    ("atan", np.arctan),
    ("atan2", np.arctan2),
    ("sinh", np.sinh),
    ("cosh", np.cosh),
    ("tanh", np.tanh),
    ("asinh", np.arcsinh),
    ("acosh", np.arccosh),
    ("atanh", np.arctanh),
    ("floor", np.floor),
    ("ceil", np.ceil),
    ("trunc", np.trunc),
    ("hypot", np.hypot),
    ("exp2", np.exp2),
    ("log2", np.log2),
    ("exp", np.exp),
    ("expm1", np.expm1),
    ("log", np.log),
    ("log10", np.log10),
    ("log1p", np.log1p),
    ("sqrt", np.sqrt),
    ("fabs", np.fabs),
]
