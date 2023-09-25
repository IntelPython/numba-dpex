#ifndef __LINEAR_SEQUENCES_HPP__
#define __LINEAR_SEQUENCES_HPP__

#include "type_utils.hpp"
#include <CL/sycl.hpp>
#include <complex>

namespace ndpxutils = ndpx::runtime::utils;

template <typename Ty> class linear_sequence_step_kernel;
template <typename Ty, typename wTy> class linear_sequence_affine_kernel;
template <typename Ty> class eye_kernel;

template <typename Ty> class LinearSequenceStepFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty step_v;

public:
    LinearSequenceStepFunctor(char *dst_p, Ty v0, Ty dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        if (ndpxutils::is_complex<Ty>::value) {
            p[i] = Ty{start_v.real() + i * step_v.real(),
                      start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue exec_q,
                                size_t nelems,
                                Ty start_v,
                                Ty step_v,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    ndpxutils::validate_type_for_device<Ty>(exec_q);
    sycl::event lin_space_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<linear_sequence_step_kernel<Ty>>(
            sycl::range<1>{nelems},
            LinearSequenceStepFunctor<Ty>(array_data, start_v, step_v));
    });

    return lin_space_step_event;
}

#endif
