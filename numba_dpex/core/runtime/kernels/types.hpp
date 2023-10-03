#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <CL/sycl.hpp>
#include <complex>
#include <exception>
#include <utility>

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
        return convert_impl<dstTy, typename srcTy::value_type>(v.real());
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

template <typename T> void validate_type_for_device(const sycl::queue &q)
{
    validate_type_for_device<T>(q.get_device());
}

template <typename Op, typename Vec, std::size_t... I>
auto vec_cast_impl(const Vec &v, std::index_sequence<I...>)
{
    return Op{v[I]...};
}

template <typename dstT,
          typename srcT,
          std::size_t N,
          typename Indices = std::make_index_sequence<N>>
auto vec_cast(const sycl::vec<srcT, N> &s)
{
    if constexpr (std::is_same_v<srcT, dstT>) {
        return s;
    }
    else {
        return vec_cast_impl<sycl::vec<dstT, N>, sycl::vec<srcT, N>>(s,
                                                                     Indices{});
    }
}

} // namespace types
} // namespace kernel
} // namespace runtime
} // namespace ndpx

#endif
