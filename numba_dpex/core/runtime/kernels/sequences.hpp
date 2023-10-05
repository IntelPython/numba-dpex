#ifndef __SEQUENCES_HPP__
#define __SEQUENCES_HPP__

#include "types.hpp"
#include <CL/sycl.hpp>
#include <complex>
#include <exception>
#include <iostream>

namespace ndpx
{
namespace runtime
{
namespace kernel
{
namespace tensor
{

template <typename T> class ndpx_sequence_step_kernel;
template <typename T, typename wT> class ndpx_affine_sequence_step_kernel;

template <typename T> class SequenceStepFunctor
{
private:
    T *p = nullptr;
    T start_v;
    T step_v;

public:
    SequenceStepFunctor(char *dst_p, T v0, T dv)
        : p(reinterpret_cast<T *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        if constexpr (ndpx::runtime::kernel::types::is_complex<T>::value) {
            p[i] = T{start_v.real() + i * step_v.real(),
                     start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

template <typename T, typename wT> class AffineSequenceStepFunctor
{
private:
    T *p = nullptr;
    T start_v;
    T end_v;
    size_t n;

public:
    AffineSequenceStepFunctor(char *dst_p, T v0, T v1, size_t den)
        : p(reinterpret_cast<T *>(dst_p)), start_v(v0), end_v(v1),
          n((den == 0) ? 1 : den)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        wT wc = wT(i) / n;
        wT w = wT(n - i) / n;
        if constexpr (ndpx::runtime::kernel::types::is_complex<T>::value) {
            using reT = typename T::value_type;
            auto _w = static_cast<reT>(w);
            auto _wc = static_cast<reT>(wc);
            auto re_comb = sycl::fma(start_v.real(), _w, reT(0));
            re_comb =
                sycl::fma(end_v.real(), _wc,
                          re_comb); // start_v.real() * _w + end_v.real() * _wc;
            auto im_comb =
                sycl::fma(start_v.imag(), _w,
                          reT(0)); // start_v.imag() * _w + end_v.imag() * _wc;
            im_comb = sycl::fma(end_v.imag(), _wc, im_comb);
            T affine_comb = T{re_comb, im_comb};
            p[i] = affine_comb;
        }
        else if constexpr (std::is_floating_point<T>::value) {
            T _w = static_cast<T>(w);
            T _wc = static_cast<T>(wc);
            auto affine_comb =
                sycl::fma(start_v, _w, T(0)); // start_v * w + end_v * wc;
            affine_comb = sycl::fma(end_v, _wc, affine_comb);
            p[i] = affine_comb;
        }
        else {
            auto affine_comb = start_v * w + end_v * wc;
            p[i] = ndpx::runtime::kernel::types::convert_impl<
                T, decltype(affine_comb)>(affine_comb);
        }
    }
};

template <typename T>
sycl::event sequence_step_kernel(sycl::queue exec_q,
                                 size_t nelems,
                                 T start_v,
                                 T step_v,
                                 char *array_data,
                                 const std::vector<sycl::event> &depends)
{
    ndpx::runtime::kernel::types::validate_type_for_device<T>(exec_q);
    sycl::event seq_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<ndpx_sequence_step_kernel<T>>(
            sycl::range<1>{nelems},
            SequenceStepFunctor<T>(array_data, start_v, step_v));
    });

    return seq_step_event;
}

template <typename T>
sycl::event affine_sequence_step_kernel(sycl::queue &exec_q,
                                        size_t nelems,
                                        T start_v,
                                        T end_v,
                                        bool include_endpoint,
                                        char *array_data,
                                        const std::vector<sycl::event> &depends)
{
    ndpx::runtime::kernel::types::validate_type_for_device<T>(exec_q);
    bool device_supports_doubles = exec_q.get_device().has(sycl::aspect::fp64);
    sycl::event affine_seq_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (device_supports_doubles) {
            cgh.parallel_for<ndpx_affine_sequence_step_kernel<T, double>>(
                sycl::range<1>{nelems},
                AffineSequenceStepFunctor<T, double>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
        else {
            cgh.parallel_for<ndpx_affine_sequence_step_kernel<T, float>>(
                sycl::range<1>{nelems},
                AffineSequenceStepFunctor<T, float>(
                    array_data, start_v, end_v,
                    (include_endpoint) ? nelems - 1 : nelems));
        }
    });

    return affine_seq_step_event;
}

template <typename T>
sycl::event sequence_step(sycl::queue &exec_q,
                          size_t nelems,
                          void *start,
                          void *step,
                          char *array_data,
                          const std::vector<sycl::event> &depends)
{
    T *start_v, *step_v;
    try {
        start_v = reinterpret_cast<T *>(start);
        step_v = reinterpret_cast<T *>(step);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    auto sequence_step_event = sequence_step_kernel<T>(
        exec_q, nelems, *start_v, *step_v, array_data, depends);

    return sequence_step_event;
}

template <typename T>
sycl::event affine_sequence_step(sycl::queue &exec_q,
                                 size_t nelems,
                                 void *start,
                                 void *end,
                                 bool include_endpoint,
                                 char *array_data,
                                 const std::vector<sycl::event> &depends)
{
    T *start_v, *end_v;
    try {
        start_v = reinterpret_cast<T *>(start);
        end_v = reinterpret_cast<T *>(end);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    auto affine_sequence_step_event =
        affine_sequence_step_kernel<T>(exec_q, nelems, *start_v, *end_v,
                                       include_endpoint, array_data, depends);

    return affine_sequence_step_event;
}

template <typename fnT, typename T> struct SequenceStepFactory
{
    fnT get()
    {
        fnT f = sequence_step<T>;
        return f;
    }
};

template <typename fnT, typename T> struct AffineSequenceStepFactory
{
    fnT get()
    {
        fnT f = affine_sequence_step<T>;
        return f;
    }
};

typedef sycl::event (*sequence_step_ptr_t)(sycl::queue &,
                                           size_t, // num_elements
                                           void *, // start_v
                                           void *, // end_v
                                           char *, // dst_data_ptr
                                           const std::vector<sycl::event> &);

typedef sycl::event (*affine_sequence_step_ptr_t)(
    sycl::queue &,
    size_t, // num_elements
    void *, // start_v
    void *, // end_v
    bool,   // include_endpoint
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

} // namespace tensor
} // namespace kernel
} // namespace runtime
} // namespace ndpx

#endif
