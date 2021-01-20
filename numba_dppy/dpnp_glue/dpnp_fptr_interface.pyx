# distutils: language = c++
# cython: language_level=3

import ctypes


cdef extern from "backend_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
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
        DPNP_FN_COPYSIGN
        DPNP_FN_CORRELATE
        DPNP_FN_COS
        DPNP_FN_COSH
        DPNP_FN_COV
        DPNP_FN_DEGREES
        DPNP_FN_DET
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
        DPNP_FN_HYPOT
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
        DPNP_FN_REMAINDER
        DPNP_FN_RECIP
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
        DPNP_FN_TAN
        DPNP_FN_TANH
        DPNP_FN_TRANSPOSE
        DPNP_FN_TRUNC
        DPNP_FN_VAR


cdef extern from "backend_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE

cdef extern from "backend_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type)


cdef DPNPFuncName get_DPNPFuncName_from_str(name):
    if name == "dpnp_dot":
        return DPNPFuncName.DPNP_FN_DOT
    elif name == "dpnp_matmul":
        return DPNPFuncName.DPNP_FN_MATMUL
    elif name == "dpnp_sum":
        return DPNPFuncName.DPNP_FN_SUM
    elif name == "dpnp_prod":
        return DPNPFuncName.DPNP_FN_PROD
    elif name == "dpnp_argmax":
        return DPNPFuncName.DPNP_FN_ARGMAX
    elif name == "dpnp_max":
        return DPNPFuncName.DPNP_FN_MAX
    elif name == "dpnp_argmin":
        return DPNPFuncName.DPNP_FN_ARGMIN
    elif name == "dpnp_min":
        return DPNPFuncName.DPNP_FN_MIN
    elif name == "dpnp_mean":
        return DPNPFuncName.DPNP_FN_MEAN
    elif name == "dpnp_median":
        return DPNPFuncName.DPNP_FN_MEDIAN
    elif name == "dpnp_argsort":
        return DPNPFuncName.DPNP_FN_ARGSORT
    elif name == "dpnp_cov":
        return DPNPFuncName.DPNP_FN_COV
    elif name == "dpnp_eig":
        return DPNPFuncName.DPNP_FN_EIG
    elif name == "dpnp_random_sample":
        return DPNPFuncName.DPNP_FN_RNG_UNIFORM
    elif name == "dpnp_beta":
        return DPNPFuncName.DPNP_FN_RNG_BETA
    elif name == "dpnp_binomial":
        return DPNPFuncName.DPNP_FN_RNG_BINOMIAL
    elif name == "dpnp_chisquare":
        return DPNPFuncName.DPNP_FN_RNG_CHISQUARE
    elif name == "dpnp_exponential":
        return DPNPFuncName.DPNP_FN_RNG_EXPONENTIAL
    elif name == "dpnp_gamma":
        return DPNPFuncName.DPNP_FN_RNG_GAMMA
    elif name == "dpnp_geometric":
        return DPNPFuncName.DPNP_FN_RNG_GEOMETRIC
    elif name == "dpnp_gumbel":
        return DPNPFuncName.DPNP_FN_RNG_GUMBEL
    elif name == "dpnp_hypergeometric":
        return DPNPFuncName.DPNP_FN_RNG_HYPERGEOMETRIC
    elif name == "dpnp_laplace":
        return DPNPFuncName.DPNP_FN_RNG_LAPLACE
    elif name == "dpnp_lognormal":
        return DPNPFuncName.DPNP_FN_RNG_LOGNORMAL
    elif name == "dpnp_multinomial":
        return DPNPFuncName.DPNP_FN_RNG_MULTINOMIAL
    elif name == "dpnp_multivariate_normal":
        return DPNPFuncName.DPNP_FN_RNG_MULTIVARIATE_NORMAL
    elif name == "dpnp_negative_binomial":
        return DPNPFuncName.DPNP_FN_RNG_NEGATIVE_BINOMIAL
    elif name == "dpnp_normal":
        return DPNPFuncName.DPNP_FN_RNG_NORMAL
    elif name == "dpnp_poisson":
        return DPNPFuncName.DPNP_FN_RNG_POISSON
    elif name == "dpnp_rayleigh":
        return DPNPFuncName.DPNP_FN_RNG_RAYLEIGH
    elif name == "dpnp_standard_cauchy":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_CAUCHY
    elif name == "dpnp_standard_exponential":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_EXPONENTIAL
    elif name == "dpnp_standard_gamma":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_GAMMA
    elif name == "dpnp_uniform":
        return DPNPFuncName.DPNP_FN_RNG_UNIFORM
    elif name == "dpnp_weibull":
        return DPNPFuncName.DPNP_FN_RNG_WEIBULL
    elif name == "dpnp_vdot":
        return DPNPFuncName.DPNP_FN_DOT
    elif name == "dpnp_cholesky":
        return DPNPFuncName.DPNP_FN_CHOLESKY
    elif name == "dpnp_det":
        return DPNPFuncName.DPNP_FN_DET
    elif name == "dpnp_matrix_rank":
        return DPNPFuncName.DPNP_FN_MATRIX_RANK
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
