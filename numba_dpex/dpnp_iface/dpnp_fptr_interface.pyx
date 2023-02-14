# SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

# distutils: language = c++
# cython: language_level=3

import ctypes


cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":  # need this namespace for Enum import
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_NONE,
        DPNP_FN_ABSOLUTE,
        DPNP_FN_ABSOLUTE_EXT,
        DPNP_FN_ADD,
        DPNP_FN_ADD_EXT,
        DPNP_FN_ALL,
        DPNP_FN_ALL_EXT,
        DPNP_FN_ALLCLOSE,
        DPNP_FN_ALLCLOSE_EXT,
        DPNP_FN_ANY,
        DPNP_FN_ANY_EXT,
        DPNP_FN_ARANGE,
        DPNP_FN_ARCCOS,
        DPNP_FN_ARCCOS_EXT,
        DPNP_FN_ARCCOSH,
        DPNP_FN_ARCCOSH_EXT,
        DPNP_FN_ARCSIN,
        DPNP_FN_ARCSIN_EXT,
        DPNP_FN_ARCSINH,
        DPNP_FN_ARCSINH_EXT,
        DPNP_FN_ARCTAN,
        DPNP_FN_ARCTAN_EXT,
        DPNP_FN_ARCTAN2,
        DPNP_FN_ARCTAN2_EXT,
        DPNP_FN_ARCTANH,
        DPNP_FN_ARCTANH_EXT,
        DPNP_FN_ARGMAX,
        DPNP_FN_ARGMAX_EXT,
        DPNP_FN_ARGMIN,
        DPNP_FN_ARGMIN_EXT,
        DPNP_FN_ARGSORT,
        DPNP_FN_ARGSORT_EXT,
        DPNP_FN_AROUND,
        DPNP_FN_AROUND_EXT,
        DPNP_FN_ASTYPE,
        DPNP_FN_ASTYPE_EXT,
        DPNP_FN_BITWISE_AND,
        DPNP_FN_BITWISE_AND_EXT,
        DPNP_FN_BITWISE_OR,
        DPNP_FN_BITWISE_OR_EXT,
        DPNP_FN_BITWISE_XOR,
        DPNP_FN_BITWISE_XOR_EXT,
        DPNP_FN_CBRT,
        DPNP_FN_CBRT_EXT,
        DPNP_FN_CEIL,
        DPNP_FN_CEIL_EXT,
        DPNP_FN_CHOLESKY,
        DPNP_FN_CHOLESKY_EXT,
        DPNP_FN_CONJIGUATE,
        DPNP_FN_CONJIGUATE_EXT,
        DPNP_FN_CHOOSE,
        DPNP_FN_CHOOSE_EXT,
        DPNP_FN_COPY,
        DPNP_FN_COPY_EXT,
        DPNP_FN_COPYSIGN,
        DPNP_FN_COPYSIGN_EXT,
        DPNP_FN_COPYTO,
        DPNP_FN_COPYTO_EXT,
        DPNP_FN_CORRELATE,
        DPNP_FN_CORRELATE_EXT,
        DPNP_FN_COS,
        DPNP_FN_COS_EXT,
        DPNP_FN_COSH,
        DPNP_FN_COSH_EXT,
        DPNP_FN_COUNT_NONZERO,
        DPNP_FN_COUNT_NONZERO_EXT,
        DPNP_FN_COV,
        DPNP_FN_COV_EXT,
        DPNP_FN_CROSS,
        DPNP_FN_CROSS_EXT,
        DPNP_FN_CUMPROD,
        DPNP_FN_CUMPROD_EXT,
        DPNP_FN_CUMSUM,
        DPNP_FN_CUMSUM_EXT,
        DPNP_FN_DEGREES,
        DPNP_FN_DEGREES_EXT,
        DPNP_FN_DET,
        DPNP_FN_DET_EXT,
        DPNP_FN_DIAG,
        DPNP_FN_DIAG_EXT,
        DPNP_FN_DIAG_INDICES,
        DPNP_FN_DIAG_INDICES_EXT,
        DPNP_FN_DIAGONAL,
        DPNP_FN_DIAGONAL_EXT,
        DPNP_FN_DIVIDE,
        DPNP_FN_DIVIDE_EXT,
        DPNP_FN_DOT,
        DPNP_FN_DOT_EXT,
        DPNP_FN_EDIFF1D,
        DPNP_FN_EDIFF1D_EXT,
        DPNP_FN_EIG,
        DPNP_FN_EIG_EXT,
        DPNP_FN_EIGVALS,
        DPNP_FN_EIGVALS_EXT,
        DPNP_FN_ERF,
        DPNP_FN_ERF_EXT,
        DPNP_FN_EYE,
        DPNP_FN_EYE_EXT,
        DPNP_FN_EXP,
        DPNP_FN_EXP_EXT,
        DPNP_FN_EXP2,
        DPNP_FN_EXP2_EXT,
        DPNP_FN_EXPM1,
        DPNP_FN_EXPM1_EXT,
        DPNP_FN_FABS,
        DPNP_FN_FABS_EXT,
        DPNP_FN_FFT_FFT,
        DPNP_FN_FFT_FFT_EXT,
        DPNP_FN_FFT_RFFT,
        DPNP_FN_FFT_RFFT_EXT,
        DPNP_FN_FILL_DIAGONAL,
        DPNP_FN_FILL_DIAGONAL_EXT,
        DPNP_FN_FLATTEN,
        DPNP_FN_FLATTEN_EXT,
        DPNP_FN_FLOOR,
        DPNP_FN_FLOOR_EXT,
        DPNP_FN_FLOOR_DIVIDE,
        DPNP_FN_FLOOR_DIVIDE_EXT,
        DPNP_FN_FMOD,
        DPNP_FN_FMOD_EXT,
        DPNP_FN_FULL,
        DPNP_FN_FULL_EXT,
        DPNP_FN_FULL_LIKE,
        DPNP_FN_FULL_LIKE_EXT,
        DPNP_FN_HYPOT,
        DPNP_FN_HYPOT_EXT,
        DPNP_FN_IDENTITY,
        DPNP_FN_IDENTITY_EXT,
        DPNP_FN_INITVAL,
        DPNP_FN_INITVAL_EXT,
        DPNP_FN_INV,
        DPNP_FN_INV_EXT,
        DPNP_FN_INVERT,
        DPNP_FN_INVERT_EXT,
        DPNP_FN_KRON,
        DPNP_FN_KRON_EXT,
        DPNP_FN_LEFT_SHIFT,
        DPNP_FN_LEFT_SHIFT_EXT,
        DPNP_FN_LOG,
        DPNP_FN_LOG_EXT,
        DPNP_FN_LOG10,
        DPNP_FN_LOG10_EXT,
        DPNP_FN_LOG2,
        DPNP_FN_LOG2_EXT,
        DPNP_FN_LOG1P,
        DPNP_FN_LOG1P_EXT,
        DPNP_FN_MATMUL,
        DPNP_FN_MATMUL_EXT,
        DPNP_FN_MATRIX_RANK,
        DPNP_FN_MATRIX_RANK_EXT,
        DPNP_FN_MAX,
        DPNP_FN_MAX_EXT,
        DPNP_FN_MAXIMUM,
        DPNP_FN_MAXIMUM_EXT,
        DPNP_FN_MEAN,
        DPNP_FN_MEAN_EXT,
        DPNP_FN_MEDIAN,
        DPNP_FN_MEDIAN_EXT,
        DPNP_FN_MIN,
        DPNP_FN_MIN_EXT,
        DPNP_FN_MINIMUM,
        DPNP_FN_MINIMUM_EXT,
        DPNP_FN_MODF,
        DPNP_FN_MODF_EXT,
        DPNP_FN_MULTIPLY,
        DPNP_FN_MULTIPLY_EXT,
        DPNP_FN_NANVAR,
        DPNP_FN_NANVAR_EXT,
        DPNP_FN_NEGATIVE,
        DPNP_FN_NEGATIVE_EXT,
        DPNP_FN_NONZERO,
        DPNP_FN_NONZERO_EXT,
        DPNP_FN_ONES,
        DPNP_FN_ONES_EXT,
        DPNP_FN_ONES_LIKE,
        DPNP_FN_ONES_LIKE_EXT,
        DPNP_FN_PARTITION,
        DPNP_FN_PARTITION_EXT,
        DPNP_FN_PLACE,
        DPNP_FN_PLACE_EXT,
        DPNP_FN_POWER,
        DPNP_FN_POWER_EXT,
        DPNP_FN_PROD,
        DPNP_FN_PROD_EXT,
        DPNP_FN_PTP,
        DPNP_FN_PTP_EXT,
        DPNP_FN_PUT,
        DPNP_FN_PUT_EXT,
        DPNP_FN_PUT_ALONG_AXIS,
        DPNP_FN_PUT_ALONG_AXIS_EXT,
        DPNP_FN_QR,
        DPNP_FN_QR_EXT,
        DPNP_FN_RADIANS,
        DPNP_FN_RADIANS_EXT,
        DPNP_FN_REMAINDER,
        DPNP_FN_REMAINDER_EXT,
        DPNP_FN_RECIP,
        DPNP_FN_RECIP_EXT,
        DPNP_FN_REPEAT,
        DPNP_FN_REPEAT_EXT,
        DPNP_FN_RIGHT_SHIFT,
        DPNP_FN_RIGHT_SHIFT_EXT,
        DPNP_FN_RNG_BETA,
        DPNP_FN_RNG_BETA_EXT,
        DPNP_FN_RNG_BINOMIAL,
        DPNP_FN_RNG_BINOMIAL_EXT,
        DPNP_FN_RNG_CHISQUARE,
        DPNP_FN_RNG_CHISQUARE_EXT,
        DPNP_FN_RNG_EXPONENTIAL,
        DPNP_FN_RNG_EXPONENTIAL_EXT,
        DPNP_FN_RNG_F,
        DPNP_FN_RNG_F_EXT,
        DPNP_FN_RNG_GAMMA,
        DPNP_FN_RNG_GAMMA_EXT,
        DPNP_FN_RNG_GAUSSIAN,
        DPNP_FN_RNG_GAUSSIAN_EXT,
        DPNP_FN_RNG_GEOMETRIC,
        DPNP_FN_RNG_GEOMETRIC_EXT,
        DPNP_FN_RNG_GUMBEL,
        DPNP_FN_RNG_GUMBEL_EXT,
        DPNP_FN_RNG_HYPERGEOMETRIC,
        DPNP_FN_RNG_HYPERGEOMETRIC_EXT,
        DPNP_FN_RNG_LAPLACE,
        DPNP_FN_RNG_LAPLACE_EXT,
        DPNP_FN_RNG_LOGISTIC,
        DPNP_FN_RNG_LOGISTIC_EXT,
        DPNP_FN_RNG_LOGNORMAL,
        DPNP_FN_RNG_LOGNORMAL_EXT,
        DPNP_FN_RNG_MULTINOMIAL,
        DPNP_FN_RNG_MULTINOMIAL_EXT,
        DPNP_FN_RNG_MULTIVARIATE_NORMAL,
        DPNP_FN_RNG_MULTIVARIATE_NORMAL_EXT,
        DPNP_FN_RNG_NEGATIVE_BINOMIAL,
        DPNP_FN_RNG_NEGATIVE_BINOMIAL_EXT,
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE,
        DPNP_FN_RNG_NONCENTRAL_CHISQUARE_EXT,
        DPNP_FN_RNG_NORMAL,
        DPNP_FN_RNG_NORMAL_EXT,
        DPNP_FN_RNG_PARETO,
        DPNP_FN_RNG_PARETO_EXT,
        DPNP_FN_RNG_POISSON,
        DPNP_FN_RNG_POISSON_EXT,
        DPNP_FN_RNG_POWER,
        DPNP_FN_RNG_POWER_EXT,
        DPNP_FN_RNG_RAYLEIGH,
        DPNP_FN_RNG_RAYLEIGH_EXT,
        DPNP_FN_RNG_SRAND,
        DPNP_FN_RNG_SRAND_EXT,
        DPNP_FN_RNG_SHUFFLE,
        DPNP_FN_RNG_SHUFFLE_EXT,
        DPNP_FN_RNG_STANDARD_CAUCHY,
        DPNP_FN_RNG_STANDARD_CAUCHY_EXT,
        DPNP_FN_RNG_STANDARD_EXPONENTIAL,
        DPNP_FN_RNG_STANDARD_EXPONENTIAL_EXT,
        DPNP_FN_RNG_STANDARD_GAMMA,
        DPNP_FN_RNG_STANDARD_GAMMA_EXT,
        DPNP_FN_RNG_STANDARD_NORMAL,
        DPNP_FN_RNG_STANDARD_T,
        DPNP_FN_RNG_STANDARD_T_EXT,
        DPNP_FN_RNG_TRIANGULAR,
        DPNP_FN_RNG_TRIANGULAR_EXT,
        DPNP_FN_RNG_UNIFORM,
        DPNP_FN_RNG_UNIFORM_EXT,
        DPNP_FN_RNG_VONMISES,
        DPNP_FN_RNG_VONMISES_EXT,
        DPNP_FN_RNG_WALD,
        DPNP_FN_RNG_WALD_EXT,
        DPNP_FN_RNG_WEIBULL,
        DPNP_FN_RNG_WEIBULL_EXT,
        DPNP_FN_RNG_ZIPF,
        DPNP_FN_RNG_ZIPF_EXT,
        DPNP_FN_SEARCHSORTED,
        DPNP_FN_SEARCHSORTED_EXT,
        DPNP_FN_SIGN,
        DPNP_FN_SIGN_EXT,
        DPNP_FN_SIN,
        DPNP_FN_SIN_EXT,
        DPNP_FN_SINH,
        DPNP_FN_SINH_EXT,
        DPNP_FN_SORT,
        DPNP_FN_SORT_EXT,
        DPNP_FN_SQRT,
        DPNP_FN_SQRT_EXT,
        DPNP_FN_SQUARE,
        DPNP_FN_SQUARE_EXT,
        DPNP_FN_STD,
        DPNP_FN_STD_EXT,
        DPNP_FN_SUBTRACT,
        DPNP_FN_SUBTRACT_EXT,
        DPNP_FN_SUM,
        DPNP_FN_SUM_EXT,
        DPNP_FN_SVD,
        DPNP_FN_SVD_EXT,
        DPNP_FN_TAKE,
        DPNP_FN_TAKE_EXT,
        DPNP_FN_TAN,
        DPNP_FN_TAN_EXT,
        DPNP_FN_TANH,
        DPNP_FN_TANH_EXT,
        DPNP_FN_TRANSPOSE,
        DPNP_FN_TRANSPOSE_EXT,
        DPNP_FN_TRACE,
        DPNP_FN_TRACE_EXT,
        DPNP_FN_TRAPZ,
        DPNP_FN_TRAPZ_EXT,
        DPNP_FN_TRI,
        DPNP_FN_TRI_EXT,
        DPNP_FN_TRIL,
        DPNP_FN_TRIL_EXT,
        DPNP_FN_TRIU,
        DPNP_FN_TRIU_EXT,
        DPNP_FN_TRUNC,
        DPNP_FN_TRUNC_EXT,
        DPNP_FN_VANDER,
        DPNP_FN_VANDER_EXT,
        DPNP_FN_VAR,
        DPNP_FN_VAR_EXT,
        DPNP_FN_ZEROS,
        DPNP_FN_ZEROS_EXT,
        DPNP_FN_ZEROS_LIKE,
        DPNP_FN_ZEROS_LIKE_EXT,
        DPNP_FN_LAST,



cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncType":  # need this namespace for Enum import
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE
        DPNP_FT_BOOL

cdef extern from "dpnp_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr

    DPNPFuncData get_dpnp_function_ptr(DPNPFuncName name, DPNPFuncType first_type, DPNPFuncType second_type) except +



cdef DPNPFuncName get_DPNPFuncName_from_str(name):
    if name == "dpnp_all":
        return DPNPFuncName.DPNP_FN_ALL
    if name == "dpnp_argmax":
        return DPNPFuncName.DPNP_FN_ARGMAX
    if name == "dpnp_argmin":
        return DPNPFuncName.DPNP_FN_ARGMIN
    if name == "dpnp_argsort":
        return DPNPFuncName.DPNP_FN_ARGSORT
    if name == "dpnp_beta":
        return DPNPFuncName.DPNP_FN_RNG_BETA
    if name == "dpnp_binomial":
        return DPNPFuncName.DPNP_FN_RNG_BINOMIAL
    if name == "dpnp_chisquare":
        return DPNPFuncName.DPNP_FN_RNG_CHISQUARE
    if name == "dpnp_cholesky":
        return DPNPFuncName.DPNP_FN_CHOLESKY
    if name == "dpnp_copy":
        return DPNPFuncName.DPNP_FN_COPY
    if name == "dpnp_cov":
        return DPNPFuncName.DPNP_FN_COV
    if name == "dpnp_cumprod":
        return DPNPFuncName.DPNP_FN_CUMPROD
    if name == "dpnp_cumsum":
        return DPNPFuncName.DPNP_FN_CUMSUM
    if name == "dpnp_cumsum_ext":
        return DPNPFuncName.DPNP_FN_CUMSUM_EXT
    if name == "dpnp_det":
        return DPNPFuncName.DPNP_FN_DET
    if name == "dpnp_diagonal":
        return DPNPFuncName.DPNP_FN_DIAGONAL
    if name == "dpnp_dot":
        return DPNPFuncName.DPNP_FN_DOT
    if name == "dpnp_eig":
        return DPNPFuncName.DPNP_FN_EIG
    if name == "dpnp_exponential":
        return DPNPFuncName.DPNP_FN_RNG_EXPONENTIAL
    if name == "dpnp_full_like":
        return DPNPFuncName.DPNP_FN_INITVAL
    if name == "dpnp_full":
        return DPNPFuncName.DPNP_FN_FULL
    if name == "dpnp_gamma":
        return DPNPFuncName.DPNP_FN_RNG_GAMMA
    if name == "dpnp_geometric":
        return DPNPFuncName.DPNP_FN_RNG_GEOMETRIC
    if name == "dpnp_gumbel":
        return DPNPFuncName.DPNP_FN_RNG_GUMBEL
    if name == "dpnp_hypergeometric":
        return DPNPFuncName.DPNP_FN_RNG_HYPERGEOMETRIC
    if name == "dpnp_laplace":
        return DPNPFuncName.DPNP_FN_RNG_LAPLACE
    if name == "dpnp_lognormal":
        return DPNPFuncName.DPNP_FN_RNG_LOGNORMAL
    if name == "dpnp_matmul":
        return DPNPFuncName.DPNP_FN_MATMUL
    if name == "dpnp_matrix_rank":
        return DPNPFuncName.DPNP_FN_MATRIX_RANK
    if name == "dpnp_max":
        return DPNPFuncName.DPNP_FN_MAX
    if name == "dpnp_mean":
        return DPNPFuncName.DPNP_FN_MEAN
    if name == "dpnp_median":
        return DPNPFuncName.DPNP_FN_MEDIAN
    if name == "dpnp_min":
        return DPNPFuncName.DPNP_FN_MIN
    if name == "dpnp_multinomial":
        return DPNPFuncName.DPNP_FN_RNG_MULTINOMIAL
    if name == "dpnp_multivariate_normal":
        return DPNPFuncName.DPNP_FN_RNG_MULTIVARIATE_NORMAL
    if name == "dpnp_negative_binomial":
        return DPNPFuncName.DPNP_FN_RNG_NEGATIVE_BINOMIAL
    if name == "dpnp_normal":
        return DPNPFuncName.DPNP_FN_RNG_NORMAL
    if name == "dpnp_ones_like":
        return DPNPFuncName.DPNP_FN_INITVAL
    if name == "dpnp_partition":
        return DPNPFuncName.DPNP_FN_PARTITION
    if name == "dpnp_poisson":
        return DPNPFuncName.DPNP_FN_RNG_POISSON
    if name == "dpnp_prod":
        return DPNPFuncName.DPNP_FN_PROD
    if name == "dpnp_random_sample":
        return DPNPFuncName.DPNP_FN_RNG_UNIFORM
    if name == "dpnp_rayleigh":
        return DPNPFuncName.DPNP_FN_RNG_RAYLEIGH
    if name == "dpnp_repeat":
        return DPNPFuncName.DPNP_FN_REPEAT
    if name == "dpnp_sort":
        return DPNPFuncName.DPNP_FN_SORT
    if name == "dpnp_standard_cauchy":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_CAUCHY
    if name == "dpnp_standard_exponential":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_EXPONENTIAL
    if name == "dpnp_standard_gamma":
        return DPNPFuncName.DPNP_FN_RNG_STANDARD_GAMMA
    if name == "dpnp_sum":
        return DPNPFuncName.DPNP_FN_SUM
    if name == "dpnp_take":
        return DPNPFuncName.DPNP_FN_TAKE
    if name == "dpnp_trace":
        return DPNPFuncName.DPNP_FN_TRACE
    if name == "dpnp_uniform":
        return DPNPFuncName.DPNP_FN_RNG_UNIFORM
    if name == "dpnp_vdot":
        return DPNPFuncName.DPNP_FN_DOT
    if name == "dpnp_weibull":
        return DPNPFuncName.DPNP_FN_RNG_WEIBULL
    if name == "dpnp_zeros_like":
        return DPNPFuncName.DPNP_FN_INITVAL

    raise ValueError("Unknown dpnp function requested: " + name.split("_")[1])


cdef DPNPFuncType get_DPNPFuncType_from_str(name):
    if name == "float32":
        return DPNPFuncType.DPNP_FT_FLOAT
    elif name == "int32" or name == "uint32":
        return DPNPFuncType.DPNP_FT_INT
    elif name == "float64":
        return DPNPFuncType.DPNP_FT_DOUBLE
    elif name == "int64" or name == "uint64":
        return DPNPFuncType.DPNP_FT_LONG
    elif name == "bool":
        return DPNPFuncType.DPNP_FT_BOOL
    else:
        return DPNPFuncType.DPNP_FT_NONE

from libc.stdint cimport uintptr_t
from libc.stdio cimport printf


cpdef get_dpnp_fn_ptr(name, types):
    cdef DPNPFuncName func_name = get_DPNPFuncName_from_str(name)
    cdef DPNPFuncType first_type = get_DPNPFuncType_from_str(types[0])
    cdef DPNPFuncType second_type = get_DPNPFuncType_from_str(types[1])

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(func_name, first_type, second_type)
    cdef uintptr_t fn_ptr = <uintptr_t>kernel_data.ptr

    return <object>fn_ptr
