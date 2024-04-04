# SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

"""Typing declarations for all ``math`` stdlib function in SPIRVTypingContext.
"""
import math

from numba.core import types
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    Registry,
    signature,
)

registry = Registry()
builtin_attr = registry.register_attr
infer_global = registry.register_global


@builtin_attr
class MathModuleAttribute(AttributeTemplate):
    key = types.Module(math)

    def resolve_fabs(self, mod):
        return types.Function(MathFabsFn)

    def resolve_exp(self, mod):
        return types.Function(MathExpFn)

    def resolve_expm1(self, mod):
        return types.Function(MathExpm1Fn)

    def resolve_sqrt(self, mod):
        return types.Function(MathSqrtFn)

    def resolve_log(self, mod):
        return types.Function(MathLogFn)

    def resolve_log1p(self, mod):
        return types.Function(MathLog1pFn)

    def resolve_log10(self, mod):
        return types.Function(MathLog10Fn)

    def resolve_sin(self, mod):
        return types.Function(MathSinFn)

    def resolve_cos(self, mod):
        return types.Function(MathCosFn)

    def resolve_tan(self, mod):
        return types.Function(MathTanFn)

    def resolve_sinh(self, mod):
        return types.Function(MathSinhFn)

    def resolve_cosh(self, mod):
        return types.Function(MathCoshFn)

    def resolve_tanh(self, mod):
        return types.Function(MathTanhFn)

    def resolve_asin(self, mod):
        return types.Function(MathAsinFn)

    def resolve_acos(self, mod):
        return types.Function(MathAcosFn)

    def resolve_atan(self, mod):
        return types.Function(MathAtanFn)

    def resolve_atan2(self, mod):
        return types.Function(MathAtan2Fn)

    def resolve_asinh(self, mod):
        return types.Function(MathAsinhFn)

    def resolve_acosh(self, mod):
        return types.Function(MathAcoshFn)

    def resolve_atanh(self, mod):
        return types.Function(MathAtanhFn)

    def resolve_pi(self, mod):
        return types.float64

    def resolve_e(self, mod):
        return types.float64

    def resolve_floor(self, mod):
        return types.Function(MathFloorFn)

    def resolve_ceil(self, mod):
        return types.Function(MathCeilFn)

    def resolve_trunc(self, mod):
        return types.Function(MathTruncFn)

    def resolve_isnan(self, mod):
        return types.Function(MathIsnanFn)

    def resolve_isinf(self, mod):
        return types.Function(MathIsinfFn)

    def resolve_degrees(self, mod):
        return types.Function(MathDegreesFn)

    def resolve_radians(self, mod):
        return types.Function(MathRadiansFn)

    def resolve_copysign(self, mod):
        return types.Function(MathCopysignFn)

    def resolve_fmod(self, mod):
        return types.Function(MathFmodFn)

    def resolve_pow(self, mod):
        return types.Function(MathPowFn)

    def resolve_erf(self, mod):
        return types.Function(MathErfFn)

    def resolve_erfc(self, mod):
        return types.Function(MathErfcFn)

    def resolve_gamma(self, mod):
        return types.Function(MathGammaFn)

    def resolve_lgamma(self, mod):
        return types.Function(MathLgammaFn)


class UnaryMathFuncTemplate(ConcreteTemplate):
    cases = [
        signature(types.float64, types.int64),
        signature(types.float64, types.uint64),
        signature(types.float32, types.float32),
        signature(types.float64, types.float64),
    ]


class MathFabsFn(UnaryMathFuncTemplate):
    key = math.fabs


class MathExpFn(UnaryMathFuncTemplate):
    key = math.exp


class MathExpm1Fn(UnaryMathFuncTemplate):
    key = math.expm1


class MathSqrtFn(UnaryMathFuncTemplate):
    key = math.sqrt


class MathLogFn(UnaryMathFuncTemplate):
    key = math.log


class MathLog1pFn(UnaryMathFuncTemplate):
    key = math.log1p


class MathLog10Fn(UnaryMathFuncTemplate):
    key = math.log10


class MathSinFn(UnaryMathFuncTemplate):
    key = math.sin


class MathCosFn(UnaryMathFuncTemplate):
    key = math.cos


class MathTanFn(UnaryMathFuncTemplate):
    key = math.tan


class MathSinhFn(UnaryMathFuncTemplate):
    key = math.sinh


class MathCoshFn(UnaryMathFuncTemplate):
    key = math.cosh


class MathTanhFn(UnaryMathFuncTemplate):
    key = math.tanh


class MathAsinFn(UnaryMathFuncTemplate):
    key = math.asin


class MathAcosFn(UnaryMathFuncTemplate):
    key = math.acos


class MathAtanFn(UnaryMathFuncTemplate):
    key = math.atan


class MathAtan2Fn(ConcreteTemplate):
    key = math.atan2
    cases = [
        signature(types.float64, types.int64, types.int64),
        signature(types.float64, types.uint64, types.uint64),
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


class MathAsinhFn(UnaryMathFuncTemplate):
    key = math.asinh


class MathAcoshFn(UnaryMathFuncTemplate):
    key = math.acosh


class MathAtanhFn(UnaryMathFuncTemplate):
    key = math.atanh


class MathFloorFn(UnaryMathFuncTemplate):
    key = math.floor


class MathCeilFn(UnaryMathFuncTemplate):
    key = math.ceil


class MathTruncFn(UnaryMathFuncTemplate):
    key = math.trunc


class MathRadiansFn(UnaryMathFuncTemplate):
    key = math.radians


class MathDegreesFn(UnaryMathFuncTemplate):
    key = math.degrees


class MathErfFn(UnaryMathFuncTemplate):
    key = math.erf


class MathErfcFn(UnaryMathFuncTemplate):
    key = math.erfc


class MathGammaFn(UnaryMathFuncTemplate):
    key = math.gamma


class MathLgammaFn(UnaryMathFuncTemplate):
    key = math.lgamma


class BinaryMathFuncTemplate(ConcreteTemplate):
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
    ]


class MathCopysignFn(BinaryMathFuncTemplate):
    key = math.copysign


class MathFmodFn(BinaryMathFuncTemplate):
    key = math.fmod


class MathPowFn(ConcreteTemplate):
    key = math.pow
    cases = [
        signature(types.float32, types.float32, types.float32),
        signature(types.float64, types.float64, types.float64),
        signature(types.float32, types.float32, types.int32),
        signature(types.float64, types.float64, types.int32),
    ]


class MathIsnanFn(ConcreteTemplate):
    key = math.isnan
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]


class MathIsinfFn(ConcreteTemplate):
    key = math.isinf
    cases = [
        signature(types.boolean, types.int64),
        signature(types.boolean, types.uint64),
        signature(types.boolean, types.float32),
        signature(types.boolean, types.float64),
    ]


infer_global(math, types.Module(math))
infer_global(math.fabs, types.Function(MathFabsFn))
infer_global(math.exp, types.Function(MathExpFn))
infer_global(math.expm1, types.Function(MathExpm1Fn))
infer_global(math.sqrt, types.Function(MathSqrtFn))
infer_global(math.log, types.Function(MathLogFn))
infer_global(math.log1p, types.Function(MathLog1pFn))
infer_global(math.log10, types.Function(MathLog10Fn))
infer_global(math.sin, types.Function(MathSinFn))
infer_global(math.cos, types.Function(MathCosFn))
infer_global(math.tan, types.Function(MathTanFn))
infer_global(math.sinh, types.Function(MathSinhFn))
infer_global(math.cosh, types.Function(MathCoshFn))
infer_global(math.tanh, types.Function(MathTanhFn))
infer_global(math.asin, types.Function(MathAsinFn))
infer_global(math.acos, types.Function(MathAcosFn))
infer_global(math.atan, types.Function(MathAtanFn))
infer_global(math.atan2, types.Function(MathAtan2Fn))
infer_global(math.asinh, types.Function(MathAsinhFn))
infer_global(math.acosh, types.Function(MathAcoshFn))
infer_global(math.atanh, types.Function(MathAtanhFn))
infer_global(math.floor, types.Function(MathFloorFn))
infer_global(math.ceil, types.Function(MathCeilFn))
infer_global(math.trunc, types.Function(MathTruncFn))
infer_global(math.isnan, types.Function(MathIsnanFn))
infer_global(math.isinf, types.Function(MathIsinfFn))
infer_global(math.degrees, types.Function(MathDegreesFn))
infer_global(math.radians, types.Function(MathRadiansFn))
infer_global(math.copysign, types.Function(MathCopysignFn))
infer_global(math.fmod, types.Function(MathFmodFn))
infer_global(math.pow, types.Function(MathPowFn))
infer_global(math.erf, types.Function(MathErfFn))
infer_global(math.erfc, types.Function(MathErfcFn))
infer_global(math.gamma, types.Function(MathGammaFn))
infer_global(math.lgamma, types.Function(MathLgammaFn))
