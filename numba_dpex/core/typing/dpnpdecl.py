# SPDX-FileCopyrightText: 2022 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp
from numba.core import types
from numba.core.typing.npydecl import (
    Numpy_rules_ufunc,
    NumpyRulesArrayOperator,
    NumpyRulesInplaceArrayOperator,
    NumpyRulesUnaryArrayOperator,
)
from numba.core.typing.templates import Registry

registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


def _install_operations(cls: NumpyRulesArrayOperator):
    for op, ufunc_name in cls._op_map.items():
        infer_global(op)(
            type(
                "DpnpRulesArrayOperator_" + ufunc_name,
                (cls,),
                dict(key=op),
            )
        )


class DpnpRulesArrayOperator(NumpyRulesArrayOperator):
    @property
    def ufunc(self):
        return getattr(dpnp, self._op_map[self.key])

    @classmethod
    def install_operations(cls):
        _install_operations(cls)


class DpnpRulesInplaceArrayOperator(NumpyRulesInplaceArrayOperator):
    @classmethod
    def install_operations(cls):
        _install_operations(cls)


class DpnpRulesUnaryArrayOperator(NumpyRulesUnaryArrayOperator):
    @classmethod
    def install_operations(cls):
        _install_operations(cls)


# list of unary ufuncs to register
_math_operations = [
    "add",
    "subtract",
    "multiply",
    "floor_divide",
    "negative",
    "power",
    "remainder",
    "fmod",
    "absolute",
    "sign",
    "conjugate",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    "divide",
    "true_divide",
    "mod",
    "abs",
    "fabs",
    "erf",
]

_trigonometric_functions = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    "degrees",
    "radians",
]

_bit_twiddling_functions = [
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    "bitwise_not",
]

_comparison_functions = [
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
]

_floating_functions = [
    "isfinite",
    "isinf",
    "isnan",
    "copysign",
    "modf",
    "frexp",
    "floor",
    "ceil",
    "trunc",
]

_logic_functions = []


# This is a set of the ufuncs that are not yet supported by Lowering. In order
# to trigger no-python mode we must not register them until their Lowering is
# implemented.
#
# It also works as a nice TODO list for ufunc support :)
_unsupported = set(
    [
        "frexp",  # Not supported by Numba
        "modf",  # Not supported by Numba
        "logaddexp",
        "logaddexp2",
        "positive",
        "float_power",
        "rint",
        "divmod",
        "gcd",
        "lcm",
        "signbit",
        "nextafter",
        "ldexp",
        "spacing",
        "isnat",
        "cbrt",
    ]
)

# TODO: A list of ufuncs that are in fact aliases of other ufuncs. They need
# to insert the resolve method, but not register the ufunc itself.
# In a meantime let's just register them as user functions:
# TODO: for some reason it affects "mod", but does not affect "bitwise_not" and
# "abs". May be mod is not an alias?
_aliases = {"bitwise_not", "abs"}

all_ufuncs = sum(
    [
        _math_operations,
        _trigonometric_functions,
        _bit_twiddling_functions,
        _comparison_functions,
        _floating_functions,
        _logic_functions,
    ],
    [],
)

supported_ufuncs = [x for x in all_ufuncs if x not in _unsupported]


def _dpnp_ufunc(name):
    func = getattr(dpnp, name)

    class typing_class(Numpy_rules_ufunc):
        key = func

    typing_class.__name__ = "resolve_{0}".format(name)

    if name not in _aliases:  # if not name in _aliases
        infer_global(func, types.Function(typing_class))


for func in supported_ufuncs:
    _dpnp_ufunc(func)


DpnpRulesArrayOperator.install_operations()
DpnpRulesInplaceArrayOperator.install_operations()
DpnpRulesUnaryArrayOperator.install_operations()
