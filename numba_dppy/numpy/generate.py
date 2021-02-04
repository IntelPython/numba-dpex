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

from jinja2 import Environment, FileSystemLoader
import argparse

parser = argparse.ArgumentParser(
    description="Generate mapping and stubs required for NumPy support in Numba_dppy"
)
parser.add_argument("-t", required=True, help="template directory")
parser.add_argument("-o", required=True, help="output directory")


numpy_func_list = [
    {"name": "sin", "impl": "ufunc"},
    {"name": "cos", "impl": "ufunc"},
    {"name": "tan", "impl": "ufunc"},
    {"name": "arcsin", "impl": "ufunc", "alt": "asin"},
    {"name": "arccos", "impl": "ufunc", "alt": "acos"},
    {"name": "arctan", "impl": "ufunc", "alt": "atan"},
    {"name": "arctan2", "impl": "ufunc", "alt": "atan2"},
    {"name": "degrees", "impl": "generic"},
    {"name": "radians", "impl": "generic"},
    {"name": "deg2rad", "impl": "generic"},
    {"name": "rad2deg", "impl": "generic"},
    {"name": "sinh", "impl": "ufunc"},
    {"name": "cosh", "impl": "ufunc"},
    {"name": "tanh", "impl": "ufunc"},
    {"name": "arcsinh", "impl": "ufunc", "alt": "asinh"},
    {"name": "arccosh", "impl": "ufunc", "alt": "acosh"},
    {"name": "arctanh", "impl": "ufunc", "alt": "atanh"},
    {"name": "floor", "impl": "ufunc"},
    {"name": "ceil", "impl": "ufunc"},
    {"name": "trunc", "impl": "ufunc"},
    {"name": "hypot", "impl": "ufunc"},
    {"name": "exp2", "impl": "ufunc"},
    {"name": "log2", "impl": "ufunc"},
    {"name": "prod", "impl": "dpnp"},
    {"name": "sum", "impl": "dpnp"},
    {"name": "nanprod", "impl": "dpnp"},
    {"name": "nansum", "impl": "dpnp"},
    {"name": "exp", "impl": "ufunc"},
    {"name": "expm1", "impl": "ufunc"},
    {"name": "log", "impl": "ufunc"},
    {"name": "log10", "impl": "ufunc"},
    {"name": "log1p", "impl": "ufunc"},
    {"name": "add", "impl": "generic"},
    {"name": "reciprocal", "impl": "generic"},
    {"name": "negative", "impl": "generic"},
    {"name": "multiply", "impl": "generic"},
    {"name": "divide", "impl": "generic"},
    {"name": "power", "impl": "generic"},
    {"name": "subtract", "impl": "generic"},
    {"name": "true_divide", "impl": "generic"},
    {"name": "fmod", "impl": "generic"},
    {"name": "mod", "impl": "generic"},
    {"name": "remainder", "impl": "generic"},
    {"name": "sqrt", "impl": "ufunc"},
    {"name": "square", "impl": "generic"},
    {"name": "absolute", "impl": "generic"},
    {"name": "abs", "impl": "generic"},
    {"name": "fabs", "impl": "ufunc"},
    {"name": "sign", "impl": "generic"},
    {"name": "maximum", "impl": "generic"},
    {"name": "minimum", "impl": "generic"},
    {"name": "fmax", "impl": "generic"},
    {"name": "fmin", "impl": "generic"},
    {"name": "median", "impl": "dpnp"},
    {"name": "mean", "impl": "generic"},
    {"name": "cov", "impl": "dpnp"},
    {"name": "argsort", "impl": "dpnp"},
    {"name": "argmax", "impl": "dpnp"},
    {"name": "argmin", "impl": "dpnp"},
    {"name": "beta", "impl": "dpnp", "nest": "random"},
    {"name": "binomial", "impl": "dpnp", "nest": "random"},
    {"name": "chisquare", "impl": "dpnp", "nest": "random"},
    {"name": "exponential", "impl": "dpnp", "nest": "random"},
    {"name": "gamma", "impl": "dpnp", "nest": "random"},
    {"name": "geometric", "impl": "dpnp", "nest": "random"},
    {"name": "gumbel", "impl": "dpnp", "nest": "random"},
    {"name": "hypergeometric", "impl": "dpnp", "nest": "random"},
    {"name": "laplace", "impl": "dpnp", "nest": "random"},
    {"name": "lognormal", "impl": "dpnp", "nest": "random"},
    {"name": "multinomial", "impl": "dpnp", "nest": "random"},
    {"name": "multivariate_normal", "impl": "dpnp", "nest": "random"},
    {"name": "negative_binomial", "impl": "dpnp", "nest": "random"},
    {"name": "normal", "impl": "dpnp", "nest": "random"},
    {"name": "poisson", "impl": "dpnp", "nest": "random"},
    {"name": "rand", "impl": "dpnp", "nest": "random"},
    {"name": "randint", "impl": "dpnp", "nest": "random"},
    {"name": "random_integers", "impl": "dpnp", "nest": "random"},
    {"name": "random_sample", "impl": "dpnp", "nest": "random"},
    {"name": "random", "impl": "dpnp", "nest": "random"},
    {"name": "ranf", "impl": "dpnp", "nest": "random"},
    {"name": "rayleigh", "impl": "dpnp", "nest": "random"},
    {"name": "sample", "impl": "dpnp", "nest": "random"},
    {"name": "standard_cauchy", "impl": "dpnp", "nest": "random"},
    {"name": "standard_exponential", "impl": "dpnp", "nest": "random"},
    {"name": "standard_gamma", "impl": "dpnp", "nest": "random"},
    {"name": "standard_normal", "impl": "dpnp", "nest": "random"},
    {"name": "uniform", "impl": "dpnp", "nest": "random"},
    {"name": "weibull", "impl": "dpnp", "nest": "random"},
    {"name": "isfinite", "impl": "generic"},
    {"name": "isinf", "impl": "generic"},
    {"name": "isnan", "impl": "generic"},
    {"name": "logical_and", "impl": "generic"},
    {"name": "logical_or", "impl": "generic"},
    {"name": "logical_not", "impl": "generic"},
    {"name": "logical_xor", "impl": "generic"},
    {"name": "greater", "impl": "generic"},
    {"name": "greater_equal", "impl": "generic"},
    {"name": "less", "impl": "generic"},
    {"name": "less_equal", "impl": "generic"},
    {"name": "equal", "impl": "generic"},
    {"name": "not_equal", "impl": "generic"},
    {"name": "dot", "dpnp": "generic"},
    {"name": "vdot", "impl": "dpnp"},
    {"name": "matmul", "impl": "dpnp"},
    {"name": "matrix_power", "impl": "dpnp", "nest": "linalg"},
    {"name": "cholesky", "impl": "dpnp", "nest": "linalg"},
    {"name": "eig", "impl": "dpnp", "nest": "linalg"},
    {"name": "eigvals", "impl": "dpnp", "nest": "linalg"},
    {"name": "det", "impl": "dpnp", "nest": "linalg"},
    {"name": "matrix_rank", "impl": "dpnp", "nest": "linalg"},
    {"name": "multi_dot", "impl": "dpnp", "nest": "linalg"},
    {"name": "bitwise_and", "impl": "generic"},
    {"name": "bitwise_or", "impl": "generic"},
    {"name": "bitwise_xor", "impl": "generic"},
    {"name": "invert", "impl": "generic"},
    {"name": "bitwise_not", "impl": "generic"},
    {"name": "left_shift", "impl": "generic"},
    {"name": "right_shift", "impl": "generic"},
    {"name": "all", "impl": "generic"},
    {"name": "any", "impl": "generic"},
    {"name": "amax", "impl": "dpnp"},
    {"name": "amin", "impl": "dpnp"},
    {"name": "argmax", "impl": "generic"},
    {"name": "argmin", "impl": "generic"},
    {"name": "argsort", "impl": "generic"},
    {"name": "dot", "impl": "dpnp"},
    {"name": "max", "impl": "dpnp"},
    {"name": "mean", "impl": "dpnp"},
    {"name": "min", "impl": "dpnp"},
    {"name": "empty", "impl": "generic"},
    {"name": "empty_like", "impl": "generic"},
    {"name": "sdones", "impl": "generic"},
    {"name": "zeros", "impl": "generic"},
    {"name": "arange", "impl": "generic"},
]


args = parser.parse_args()
template_dir = args.t + "/"
output_dir = args.o + "/"

file_loader = FileSystemLoader(template_dir)
env = Environment(loader=file_loader, trim_blocks=True, lstrip_blocks=True)

stubs_template = env.get_template("generate_stubs.tm")
rewrite_function_name_map_template = env.get_template(
    "generate_rewrite_function_name_map.tm"
)
ufunc_template = env.get_template("generate_ufunc.tm")

func_list = numpy_func_list

generated_stubs = stubs_template.render(numpy_supported_funcs=func_list)
generated_rewrite_function_name_map = rewrite_function_name_map_template.render(
    numpy_supported_funcs=func_list
)
generated_ufunc = ufunc_template.render(numpy_supported_funcs=func_list)


with open(output_dir + "stubs.py", "w") as f:
    f.write(generated_stubs)

with open(output_dir + "maps.py", "w") as f:
    f.write(generated_rewrite_function_name_map)

with open(output_dir + "maps.py", "a") as f:
    f.write(generated_ufunc)
