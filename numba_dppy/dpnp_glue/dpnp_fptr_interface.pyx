# distutils: language = c++
# cython: language_level=3

import ctypes


cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_ABSOLUTE
        DPNP_FN_ADD
        DPNP_FN_ARANGE
        DPNP_FN_ARCCOS
        DPNP_FN_ARCCOSH
        DPNP_FN_ARCSIN
        DPNP_FN_ARCSINH
        DPNP_FN_ARCTAN
        DPNP_FN_ARCTAN2
        DPNP_FN_ARCTANH
        DPNP_FN_ARGMAX
        DPNP_FN_ARGMIN
        DPNP_FN_ARGSORT
        DPNP_FN_BITWISE_AND
        DPNP_FN_BITWISE_OR
        DPNP_FN_BITWISE_XOR
        DPNP_FN_CBRT
        DPNP_FN_CEIL
        DPNP_FN_CHOLESKY
        DPNP_FN_COPY
        DPNP_FN_COPYSIGN
        DPNP_FN_CORRELATE
        DPNP_FN_COS
        DPNP_FN_COSH
        DPNP_FN_COV
        DPNP_FN_CUMPROD
        DPNP_FN_CUMSUM
        DPNP_FN_DEGREES
        DPNP_FN_DET
        DPNP_FN_DIAGONAL
        DPNP_FN_DIVIDE
        DPNP_FN_DOT
        DPNP_FN_EIG
        DPNP_FN_EIGVALS
        DPNP_FN_EXP
        DPNP_FN_EXP2
        DPNP_FN_EXPM1
        DPNP_FN_FABS
        DPNP_FN_FFT_FFT
        DPNP_FN_FLOOR
        DPNP_FN_FLOOR_DIVIDE
        DPNP_FN_FMOD
        DPNP_FN_FULL
        DPNP_FN_HYPOT
        DPNP_FN_INITVAL
        DPNP_FN_INVERT
        DPNP_FN_LEFT_SHIFT
        DPNP_FN_LOG
        DPNP_FN_LOG10
        DPNP_FN_LOG1P
        DPNP_FN_LOG2
        DPNP_FN_MATMUL
        DPNP_FN_MATRIX_RANK
        DPNP_FN_MAX
        DPNP_FN_MAXIMUM
        DPNP_FN_MEAN
        DPNP_FN_MEDIAN
        DPNP_FN_MIN
        DPNP_FN_MINIMUM
        DPNP_FN_MODF
        DPNP_FN_MULTIPLY
        DPNP_FN_POWER
        DPNP_FN_PROD
        DPNP_FN_RADIANS
        DPNP_FN_RECIP
        DPNP_FN_REMAINDER
        DPNP_FN_RIGHT_SHIFT
        DPNP_FN_RNG_BETA
        DPNP_FN_RNG_BINOMIAL
        DPNP_FN_RNG_CHISQUARE
        DPNP_FN_RNG_EXPONENTIAL
        DPNP_FN_RNG_GAMMA
        DPNP_FN_RNG_GAUSSIAN
        DPNP_FN_RNG_GEOMETRIC
        DPNP_FN_RNG_GUMBEL
        DPNP_FN_RNG_HYPERGEOMETRIC
        DPNP_FN_RNG_LAPLACE
        DPNP_FN_RNG_LOGNORMAL
        DPNP_FN_RNG_MULTINOMIAL
        DPNP_FN_RNG_MULTIVARIATE_NORMAL
        DPNP_FN_RNG_NEGATIVE_BINOMIAL
        DPNP_FN_RNG_NORMAL
        DPNP_FN_RNG_POISSON
        DPNP_FN_RNG_RAYLEIGH
        DPNP_FN_RNG_STANDARD_CAUCHY
        DPNP_FN_RNG_STANDARD_EXPONENTIAL
        DPNP_FN_RNG_STANDARD_GAMMA
        DPNP_FN_RNG_STANDARD_NORMAL
        DPNP_FN_RNG_UNIFORM
        DPNP_FN_RNG_WEIBULL
        DPNP_FN_SIGN
        DPNP_FN_SIN
        DPNP_FN_SINH
        DPNP_FN_SORT
        DPNP_FN_SQRT
        DPNP_FN_SQUARE
        DPNP_FN_STD
        DPNP_FN_SUBTRACT
        DPNP_FN_SUM
        DPNP_FN_TAKE
        DPNP_FN_TAN
        DPNP_FN_TANH
        DPNP_FN_TRACE
        DPNP_FN_TRANSPOSE
        DPNP_FN_TRUNC
        DPNP_FN_VAR


cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE

cdef extern from "dpnp_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type)


_DPNPFuncName_from_str = {
    "dpnp_dot": DPNPFuncName.DPNP_FN_DOT,
    "dpnp_matmul": DPNPFuncName.DPNP_FN_MATMUL,
    "dpnp_sum": DPNPFuncName.DPNP_FN_SUM,
    "dpnp_prod": DPNPFuncName.DPNP_FN_PROD,
    "dpnp_argmax": DPNPFuncName.DPNP_FN_ARGMAX,
    "dpnp_max": DPNPFuncName.DPNP_FN_MAX,
    "dpnp_argmin": DPNPFuncName.DPNP_FN_ARGMIN,
    "dpnp_min": DPNPFuncName.DPNP_FN_MIN,
    "dpnp_mean": DPNPFuncName.DPNP_FN_MEAN,
    "dpnp_median": DPNPFuncName.DPNP_FN_MEDIAN,
    "dpnp_argsort": DPNPFuncName.DPNP_FN_ARGSORT,
    "dpnp_cov": DPNPFuncName.DPNP_FN_COV,
    "dpnp_eig": DPNPFuncName.DPNP_FN_EIG,
    "dpnp_random_sample": DPNPFuncName.DPNP_FN_RNG_UNIFORM,
    "dpnp_beta": DPNPFuncName.DPNP_FN_RNG_BETA,
    "dpnp_binomial": DPNPFuncName.DPNP_FN_RNG_BINOMIAL,
    "dpnp_chisquare": DPNPFuncName.DPNP_FN_RNG_CHISQUARE,
    "dpnp_exponential": DPNPFuncName.DPNP_FN_RNG_EXPONENTIAL,
    "dpnp_gamma": DPNPFuncName.DPNP_FN_RNG_GAMMA,
    "dpnp_geometric": DPNPFuncName.DPNP_FN_RNG_GEOMETRIC,
    "dpnp_gumbel": DPNPFuncName.DPNP_FN_RNG_GUMBEL,
    "dpnp_hypergeometric": DPNPFuncName.DPNP_FN_RNG_HYPERGEOMETRIC,
    "dpnp_laplace": DPNPFuncName.DPNP_FN_RNG_LAPLACE,
    "dpnp_lognormal": DPNPFuncName.DPNP_FN_RNG_LOGNORMAL,
    "dpnp_multinomial": DPNPFuncName.DPNP_FN_RNG_MULTINOMIAL,
    "dpnp_multivariate_normal": DPNPFuncName.DPNP_FN_RNG_MULTIVARIATE_NORMAL,
    "dpnp_negative_binomial": DPNPFuncName.DPNP_FN_RNG_NEGATIVE_BINOMIAL,
    "dpnp_normal": DPNPFuncName.DPNP_FN_RNG_NORMAL,
    "dpnp_poisson": DPNPFuncName.DPNP_FN_RNG_POISSON,
    "dpnp_rayleigh": DPNPFuncName.DPNP_FN_RNG_RAYLEIGH,
    "dpnp_standard_cauchy": DPNPFuncName.DPNP_FN_RNG_STANDARD_CAUCHY,
    "dpnp_standard_exponential": DPNPFuncName.DPNP_FN_RNG_STANDARD_EXPONENTIAL,
    "dpnp_standard_gamma": DPNPFuncName.DPNP_FN_RNG_STANDARD_GAMMA,
    "dpnp_uniform": DPNPFuncName.DPNP_FN_RNG_UNIFORM,
    "dpnp_weibull": DPNPFuncName.DPNP_FN_RNG_WEIBULL,
    "dpnp_vdot": DPNPFuncName.DPNP_FN_DOT,
    "dpnp_cholesky": DPNPFuncName.DPNP_FN_CHOLESKY,
    "dpnp_det": DPNPFuncName.DPNP_FN_DET,
    "dpnp_matrix_rank": DPNPFuncName.DPNP_FN_MATRIX_RANK,
    "dpnp_full": DPNPFuncName.DPNP_FN_FULL,
    "dpnp_ones_like": DPNPFuncName.DPNP_FN_INITVAL,
    "dpnp_zeros_like": DPNPFuncName.DPNP_FN_INITVAL,
    "dpnp_full_like": DPNPFuncName.DPNP_FN_INITVAL,
    "dpnp_cumsum": DPNPFuncName.DPNP_FN_CUMSUM,
    "dpnp_cumprod": DPNPFuncName.DPNP_FN_CUMPROD,
    "dpnp_sort": DPNPFuncName.DPNP_FN_SORT,
    "dpnp_copy": DPNPFuncName.DPNP_FN_COPY,
    "dpnp_take": DPNPFuncName.DPNP_FN_TAKE,
    "dpnp_trace": DPNPFuncName.DPNP_FN_TRACE,
    "dpnp_diagonal": DPNPFuncName.DPNP_FN_DIAGONAL,
}

cdef DPNPFuncName get_DPNPFuncName_from_str(name):
    if name in _DPNPFuncName_from_str:
        return _DPNPFuncName_from_str[name]
    else:
        raise ValueError("Unknown dpnp function requested: " + name.split("_")[1])


cdef DPNPFuncType get_DPNPFuncType_from_str(name):
    if name == "float32":
        return DPNPFuncType.DPNP_FT_FLOAT
    elif name == "int32" or name == "uint32" or name == "bool":
        return DPNPFuncType.DPNP_FT_INT
    elif name == "float64":
        return DPNPFuncType.DPNP_FT_DOUBLE
    elif name == "int64" or name == "uint64":
        return DPNPFuncType.DPNP_FT_LONG
    else:
        return DPNPFuncType.DPNP_FT_NONE

from libc.stdio cimport printf
from libc.stdint cimport uintptr_t

cpdef get_dpnp_fn_ptr(name, types):
    cdef DPNPFuncName func_name = get_DPNPFuncName_from_str(name)
    cdef DPNPFuncType first_type = get_DPNPFuncType_from_str(types[0])
    cdef DPNPFuncType second_type = get_DPNPFuncType_from_str(types[1])

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(func_name, first_type, second_type)
    cdef uintptr_t fn_ptr = <uintptr_t>kernel_data.ptr

    return <object>fn_ptr
