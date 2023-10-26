#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <cstdlib>
#include <complex>
#include <exception>
#include <utility>
#include <string>
#include <cxxabi.h>
#include <CL/sycl.hpp>

namespace ndpx
{
namespace runtime
{
namespace kernel
{
namespace types
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
        std::cout
            << "ndpx::runtime::kernel::types::validate_type_for_device(): here0"
            << std::endl;
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'float64'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::cout
            << "ndpx::runtime::kernel::types::validate_type_for_device(): here1"
            << std::endl;
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'complex128'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        std::cout
            << "ndpx::runtime::kernel::types::validate_type_for_device(): here2"
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

// template <typename Op, typename Vec, std::size_t... I>
// auto vec_cast_impl(const Vec &v, std::index_sequence<I...>)
// {
//     return Op{v[I]...};
// }

// template <typename dstT,
//           typename srcT,
//           std::size_t N,
//           typename Indices = std::make_index_sequence<N>>
// auto vec_cast(const sycl::vec<srcT, N> &s)
// {
//     if constexpr (std::is_same_v<srcT, dstT>) {
//         return s;
//     }
//     else {
//         return vec_cast_impl<sycl::vec<dstT, N>, sycl::vec<srcT, N>>(s,
//                                                                      Indices{});
//     }
// }

// struct usm_ndarray_types
// {
//     int typenum_to_lookup_id(int typenum) const
//     {
//         // using typenum_t = ::dpctl::tensor::type_dispatch::typenum_t;
//         auto const &api = ::dpctl::detail::dpctl_capi::get();

//         if (typenum == api.UAR_DOUBLE_) {
//             return static_cast<int>(typenum_t::DOUBLE);
//         }
//         else if (typenum == api.UAR_INT64_) {
//             return static_cast<int>(typenum_t::INT64);
//         }
//         else if (typenum == api.UAR_INT32_) {
//             return static_cast<int>(typenum_t::INT32);
//         }
//         else if (typenum == api.UAR_BOOL_) {
//             return static_cast<int>(typenum_t::BOOL);
//         }
//         else if (typenum == api.UAR_CDOUBLE_) {
//             return static_cast<int>(typenum_t::CDOUBLE);
//         }
//         else if (typenum == api.UAR_FLOAT_) {
//             return static_cast<int>(typenum_t::FLOAT);
//         }
//         else if (typenum == api.UAR_INT16_) {
//             return static_cast<int>(typenum_t::INT16);
//         }
//         else if (typenum == api.UAR_INT8_) {
//             return static_cast<int>(typenum_t::INT8);
//         }
//         else if (typenum == api.UAR_UINT64_) {
//             return static_cast<int>(typenum_t::UINT64);
//         }
//         else if (typenum == api.UAR_UINT32_) {
//             return static_cast<int>(typenum_t::UINT32);
//         }
//         else if (typenum == api.UAR_UINT16_) {
//             return static_cast<int>(typenum_t::UINT16);
//         }
//         else if (typenum == api.UAR_UINT8_) {
//             return static_cast<int>(typenum_t::UINT8);
//         }
//         else if (typenum == api.UAR_CFLOAT_) {
//             return static_cast<int>(typenum_t::CFLOAT);
//         }
//         else if (typenum == api.UAR_HALF_) {
//             return static_cast<int>(typenum_t::HALF);
//         }
//         else if (typenum == api.UAR_INT_ || typenum == api.UAR_UINT_) {
//             switch (sizeof(int)) {
//             case sizeof(int32_t):
//                 return ((typenum == api.UAR_INT_)
//                             ? static_cast<int>(typenum_t::INT32)
//                             : static_cast<int>(typenum_t::UINT32));
//             case sizeof(int64_t):
//                 return ((typenum == api.UAR_INT_)
//                             ? static_cast<int>(typenum_t::INT64)
//                             : static_cast<int>(typenum_t::UINT64));
//             default:
//                 throw_unrecognized_typenum_error(typenum);
//             }
//         }
//         else if (typenum == api.UAR_LONGLONG_ || typenum ==
//         api.UAR_ULONGLONG_)
//         {
//             switch (sizeof(long long)) {
//             case sizeof(int64_t):
//                 return ((typenum == api.UAR_LONGLONG_)
//                             ? static_cast<int>(typenum_t::INT64)
//                             : static_cast<int>(typenum_t::UINT64));
//             default:
//                 throw_unrecognized_typenum_error(typenum);
//             }
//         }
//         else {
//             throw_unrecognized_typenum_error(typenum);
//         }
//         // return code signalling error, should never be reached
//         assert(false);
//         return -1;
//     }

// private:
//     void throw_unrecognized_typenum_error(int typenum) const
//     {
//         throw std::runtime_error("Unrecognized typenum " +
//                                  std::to_string(typenum) + " encountered.");
//     }
// };

} // namespace types
} // namespace kernel
} // namespace runtime
} // namespace ndpx

#endif
