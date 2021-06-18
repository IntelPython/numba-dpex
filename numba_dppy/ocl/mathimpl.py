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

import math
import numpy
import warnings

from numba.core.imputils import Registry
from numba.core import types
from numba.core.itanium_mangler import mangle
from .oclimpl import _declare_function

registry = Registry()
lower = registry.lower

# -----------------------------------------------------------------------------

_unary_b_f = types.int32(types.float32)
_unary_b_d = types.int32(types.float64)
_unary_f_f = types.float32(types.float32)
_unary_d_d = types.float64(types.float64)
_binary_f_ff = types.float32(types.float32, types.float32)
_binary_d_dd = types.float64(types.float64, types.float64)

_binary_f_fi = types.float32(types.float32, types.int32)
_binary_f_ii = types.float32(types.int32, types.int32)
_binary_f_fl = types.float32(types.float32, types.int64)
_binary_f_fd = types.float32(types.float32, types.float64)
_binary_d_di = types.float64(types.float64, types.int32)
_binary_d_dl = types.float64(types.float64, types.int64)

sig_mapper = {
    "f->f": _unary_f_f,
    "d->d": _unary_d_d,
    "ff->f": _binary_f_ff,
    "dd->d": _binary_d_dd,
    "fi->f": _binary_f_fi,
    "fl->f": _binary_f_fl,
    "ff->f": _binary_f_ff,
    "di->d": _binary_d_di,
    "dl->d": _binary_d_dl,
    "dd->d": _binary_d_dd,
}

function_descriptors = {
    "isnan": (_unary_b_f, _unary_b_d),
    "isinf": (_unary_b_f, _unary_b_d),
    "ceil": (_unary_f_f, _unary_d_d),
    "floor": (_unary_f_f, _unary_d_d),
    "trunc": (_unary_f_f, _unary_d_d),
    "fabs": (_unary_f_f, _unary_d_d),
    "sqrt": (_unary_f_f, _unary_d_d),
    "exp": (_unary_f_f, _unary_d_d),
    "expm1": (_unary_f_f, _unary_d_d),
    "log": (_unary_f_f, _unary_d_d),
    "log10": (_unary_f_f, _unary_d_d),
    "log1p": (_unary_f_f, _unary_d_d),
    "sin": (_unary_f_f, _unary_d_d),
    "cos": (_unary_f_f, _unary_d_d),
    "tan": (_unary_f_f, _unary_d_d),
    "asin": (_unary_f_f, _unary_d_d),
    "acos": (_unary_f_f, _unary_d_d),
    "atan": (_unary_f_f, _unary_d_d),
    "sinh": (_unary_f_f, _unary_d_d),
    "cosh": (_unary_f_f, _unary_d_d),
    "tanh": (_unary_f_f, _unary_d_d),
    "asinh": (_unary_f_f, _unary_d_d),
    "acosh": (_unary_f_f, _unary_d_d),
    "atanh": (_unary_f_f, _unary_d_d),
    "copysign": (_binary_f_ff, _binary_d_dd),
    "atan2": (_binary_f_ff, _binary_d_dd),
    "pow": (_binary_f_ff, _binary_d_dd),
    "fmod": (_binary_f_ff, _binary_d_dd),
    "erf": (_unary_f_f, _unary_d_d),
    "erfc": (_unary_f_f, _unary_d_d),
    "gamma": (_unary_f_f, _unary_d_d),
    "lgamma": (_unary_f_f, _unary_d_d),
    "ldexp": (_binary_f_fi, _binary_f_fl, _binary_d_di, _binary_d_dl),
    "hypot": (_binary_f_fi, _binary_f_ff, _binary_d_dl, _binary_d_dd),
    "exp2": (_unary_f_f, _unary_d_d),
    "log2": (_unary_f_f, _unary_d_d),
    # unsupported functions listed in the math module documentation:
    # frexp, ldexp, trunc, modf, factorial, fsum
}


# some functions may be named differently by the underlying math
# library as oposed to the Python name.
_lib_counterpart = {"gamma": "tgamma"}


def _mk_fn_decl(name, decl_sig):
    sym = _lib_counterpart.get(name, name)

    def core(context, builder, sig, args):
        fn = _declare_function(
            context, builder, sym, decl_sig, decl_sig.args, mangler=mangle
        )
        res = builder.call(fn, args)
        return context.cast(builder, res, decl_sig.return_type, sig.return_type)

    core.__name__ = name
    return core


_supported = [
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "isnan",
    "isinf",
    "ceil",
    "floor",
    "fabs",
    "sqrt",
    "exp",
    "expm1",
    "log",
    "log10",
    "log1p",
    "copysign",
    "pow",
    "fmod",
    "erf",
    "erfc",
    "gamma",
    "lgamma",
    "ldexp",
    "trunc",
    "hypot",
    "exp2",
    "log2",
]


lower_ocl_impl = dict()


def function_name_to_supported_decl(name, sig):
    try:
        # only symbols present in the math module
        key = getattr(math, name)
    except AttributeError:
        try:
            key = getattr(numpy, name)
        except:
            return None

    fn = _mk_fn_decl(name, sig)
    # lower(key, *sig.args)(fn)
    lower_ocl_impl[(name, sig)] = lower(key, *sig.args)(fn)


for name in _supported:
    sigs = function_descriptors.get(name)
    if sigs is None:
        warnings.warn("OCL - failed to register '{0}'".format(name))
        continue

    for sig in sigs:
        function_name_to_supported_decl(name, sig)
