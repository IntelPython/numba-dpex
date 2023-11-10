// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file typeutils.hpp
/// \brief Provides different utility functions to handle type mechansim.
///
/// This file contains different utility functions for type handling mechanism.
/// The routines would range from handling types pertaining to SYCL, ``numba``,
/// and python.
///
//===----------------------------------------------------------------------===//

#ifndef __TYPEUTILS_HPP__
#define __TYPEUTILS_HPP__

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#if defined(__linux__) || defined(__unix__) || defined(_POSIX_VERSION)
#include <cxxabi.h> // this is gcc specific, not supported on windows
#endif
#include <CL/sycl.hpp>

namespace dpex
{
namespace rt
{
namespace kernel
{
namespace tensor
{
namespace typeutils
{

/**
 * \brief A template to return false when the type is not ``std::complex``
 *
 * \tparam T The template parameter, type to be checked.
 */
template <class T> struct is_complex : public std::false_type
{
};

/**
 * \brief A template function to check if a type is ``std::complex``.
 *
 * \tparam T The template parameter, type to be checked.
 */
template <class T> struct is_complex<std::complex<T>> : public std::true_type
{
};

/**
 * \brief An enum class to store different type ids.
 *
 * This is used to index different functions in the dispatch vector.
 */
enum class typenum_t : int
{
    BOOL = 0,
    INT8, // 1
    UINT8,
    INT16,
    UINT16,
    INT32, // 5
    UINT32,
    INT64,
    UINT64,
    HALF,
    FLOAT, // 10
    DOUBLE,
    CFLOAT,
    CDOUBLE, // 13
};

constexpr int num_types = 14; // number of elements in typenum_t

#if defined(__linux__) || defined(__unix__) || defined(_POSIX_VERSION)
/**
 * \brief Demangling a template parameter.
 *
 * This function is for debugging purposes.
 *
 * \tparam T            The template parameter to be demangled.
 * \return std::string  The ``std::string`` representation of the instantiated
 *                      template parameter.
 */
template <typename T> std::string demangle()
{
    char const *mangled = typeid(T).name();
    char *c_demangled;
    int status = 0;
    c_demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);

    std::string res;
    if (c_demangled) {
        res = c_demangled;
        free(c_demangled);
    }
    else {
        res = mangled;
        free(c_demangled);
    }
    return res;
}
#endif

/**
 * \brief Caste a value to the type in ``typenum_t`` using an ``int`` index
 *
 * This function can caste a value to the type in ``typenum_t``
 * using an integer index. E.g. ``caste_using_typeid(x,7)`` will caste ``x``
 * into ``int64_t``.
 *
 * \param value         The value to be casted.
 * \param _typeid       The index to the type specified in ``typenum_t``.
 * \return std::string  The value will be casted and then returned as an
 *                      ``std::string``.
 */
std::string caste_using_typeid(void *value, int _typeid)
{
    switch (_typeid) {
    case 0:
        return std::to_string(*(reinterpret_cast<bool *>(value)));
    case 1:
        return std::to_string(*(reinterpret_cast<int8_t *>(value)));
    case 2:
        return std::to_string(*(reinterpret_cast<uint8_t *>(value)));
    case 3:
        return std::to_string(*(reinterpret_cast<int16_t *>(value)));
    case 4:
        return std::to_string(*(reinterpret_cast<uint16_t *>(value)));
    case 5:
        return std::to_string(*(reinterpret_cast<int32_t *>(value)));
    case 6:
        return std::to_string(*(reinterpret_cast<uint32_t *>(value)));
    case 7:
        return std::to_string(*(reinterpret_cast<int64_t *>(value)));
    case 8:
        return std::to_string(*(reinterpret_cast<uint64_t *>(value)));
    case 9:
        return std::to_string(*(reinterpret_cast<sycl::half *>(value)));
    case 10:
        return std::to_string(*(reinterpret_cast<float *>(value)));
    case 11:
        return std::to_string(*(reinterpret_cast<double *>(value)));
    default:
        throw std::runtime_error(std::to_string(_typeid) +
                                 " could't be mapped to valid data type.");
    }
}

/**
 * \brief Converts a non non-complex data type into ``std::complex``
 *
 * It also does the reverse conversion.
 *
 * \tparam dstTy    The template paramter for the destination type.
 * \tparam srcTy    The template paramter for the source type.
 * \param v         The value to be converted.
 * \return dstTy    The value `v` casted to `dstTy`.
 */
template <typename dstTy, typename srcTy> dstTy convert_impl(const srcTy &v)
{
    if constexpr (std::is_same<dstTy, srcTy>::value) {
        return v;
    }
    else if constexpr (std::is_same_v<dstTy, bool> && is_complex<srcTy>::value)
    {
        // bool(complex_v) == (complex_v.real() != 0) && (complex_v.imag() !=0)
        return (convert_impl<bool, typename srcTy::value_type>(v.real()) ||
                convert_impl<bool, typename srcTy::value_type>(v.imag()));
    }
    else if constexpr (is_complex<srcTy>::value && !is_complex<dstTy>::value) {
        // real_t(complex_v) == real_t(complex_v.real())
        return (convert_impl<dstTy, typename srcTy::value_type>(v.real()));
    }
    else if constexpr (!std::is_integral<srcTy>::value &&
                       !std::is_same<dstTy, bool>::value &&
                       std::is_integral<dstTy>::value &&
                       std::is_unsigned<dstTy>::value)
    {
        // first cast to signed variant, the cast to unsigned one
        using signedT = typename std::make_signed<dstTy>::type;
        return static_cast<dstTy>(convert_impl<signedT, srcTy>(v));
    }
    else {
        return static_cast<dstTy>(v);
    }
}

/**
 * \brief Checks if a type ``T`` is supported on a device.
 *
 * \tparam T    The template parameter for the type.
 * \param d     The ``sycl::device`` to be checked on.
 */
template <typename T> void validate_type_for_device(const sycl::device &d)
{
    if constexpr (std::is_same_v<T, double>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float64'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'complex128'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        if (!d.has(sycl::aspect::fp16)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float16'");
        }
    }
}

/**
 * \brief Checks if a type `T` is supported by a SYCL queue.
 *
 * \tparam T    The template parameter for the type.
 * \param q     The ``sycl::queue`` to be checked on.
 */
template <typename T> void validate_type_for_device(const sycl::queue &q)
{
    validate_type_for_device<T>(q.get_device());
}

} // namespace typeutils
} // namespace tensor
} // namespace kernel
} // namespace rt
} // namespace dpex

#endif
