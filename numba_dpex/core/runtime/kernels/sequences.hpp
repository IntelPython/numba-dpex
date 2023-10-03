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

template <typename Ty> class sequence_step_kernel;

template <typename Ty> class SequenceStepFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty step_v;

public:
    SequenceStepFunctor(char *dst_p, Ty v0, Ty dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        if constexpr (ndpx::runtime::kernel::types::is_complex<Ty>::value) {
            p[i] = Ty{start_v.real() + i * step_v.real(),
                      start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

template <typename Ty>
sycl::event sequence_step_specialized(sycl::queue exec_q,
                                      size_t nelems,
                                      Ty start_v,
                                      Ty step_v,
                                      char *array_data,
                                      const std::vector<sycl::event> &depends)
{
    ndpx::runtime::kernel::types::validate_type_for_device<Ty>(exec_q);
    sycl::event seq_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<sequence_step_kernel<Ty>>(
            sycl::range<1>{nelems},
            SequenceStepFunctor<Ty>(array_data, start_v, step_v));
    });

    return seq_step_event;
}

template <typename Ty>
sycl::event sequence_step_opaque(sycl::queue &exec_q,
                                 size_t nelems,
                                 void *start,
                                 void *step,
                                 char *array_data,
                                 const std::vector<sycl::event> &depends)
{
    Ty *start_v;
    Ty *step_v;
    try {
        start_v = reinterpret_cast<Ty *>(start);
        step_v = reinterpret_cast<Ty *>(step);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    auto sequence_step_event = sequence_step_specialized<Ty>(
        exec_q, nelems, *start_v, *step_v, array_data, depends);

    return sequence_step_event;
}

template <typename fnT, typename Ty> struct SequenceStepFactory
{
    fnT get()
    {
        fnT f = sequence_step_opaque<Ty>;
        return f;
    }
};

typedef sycl::event (*sequence_step_opaque_ptr_t)(
    sycl::queue &,
    size_t, // num_elements
    void *, // start_v
    void *, // end_v
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

extern void init_sequence_dispatch_vectors(void);

} // namespace tensor
} // namespace kernel
} // namespace runtime
} // namespace ndpx

#endif
