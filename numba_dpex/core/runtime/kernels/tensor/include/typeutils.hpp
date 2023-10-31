// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <cstdlib>
#include <complex>
#include <exception>
#include <utility>
#include <string>
#include <cxxabi.h>
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

template <class T> struct is_complex : public std::false_type
{
};

template <class T> struct is_complex<std::complex<T>> : public std::true_type
{
};

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

std::string caste_using_typeid(void *value, int _typeid)
{
    switch (_typeid) {
    case 0:
        return std::to_string(*(reinterpret_cast<bool *>(value)));
    case 1:
        return std::to_string(*(reinterpret_cast<int8_t *>(value)));
    case 2:
        return std::to_string(*(reinterpret_cast<u_int8_t *>(value)));
    case 3:
        return std::to_string(*(reinterpret_cast<int16_t *>(value)));
    case 4:
        return std::to_string(*(reinterpret_cast<u_int16_t *>(value)));
    case 5:
        return std::to_string(*(reinterpret_cast<int32_t *>(value)));
    case 6:
        return std::to_string(*(reinterpret_cast<u_int32_t *>(value)));
    case 7:
        return std::to_string(*(reinterpret_cast<int64_t *>(value)));
    case 8:
        return std::to_string(*(reinterpret_cast<u_int64_t *>(value)));
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

template <typename T> void validate_type_for_device(const sycl::device &d)
{
    if constexpr (std::is_same_v<T, double>) {
        std::cout << "dpex::rt::kernel::tensor::typeutils::validate_type_for_"
                     "device(): here0"
                  << std::endl;
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float64'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::cout << "dpex::rt::kernel::tensor::typeutils::validate_type_for_"
                     "device(): here1"
                  << std::endl;
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'complex128'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        std::cout << "dpex::rt::kernel::tensor::typeutils::validate_type_for_"
                     "device(): here2"
                  << std::endl;
        if (!d.has(sycl::aspect::fp16)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float16'");
        }
    }
}

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
