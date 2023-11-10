// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file intervals.hpp
/// \brief Implements different tensor construction functionality.
///
/// This file contains APIs to facilitate different tensor construction and
/// manipulation routines. Namely, different SYCL kernels and related functions,
/// classes and constructs to support those kernels.
///
//===----------------------------------------------------------------------===//

#ifndef __INTERVALS_HPP__
#define __INTERVALS_HPP__

#include <iostream>
#include <exception>
#include <complex>
#include <typeinfo>

#include <Python.h>
#include <numpy/npy_common.h>
#include <numba/_arraystruct.h>

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#include "typeutils.hpp"

// Shorthand namespace for easy coding.
namespace dpexrt_tensor = dpex::rt::kernel::tensor;

namespace dpex
{
namespace rt
{
namespace kernel
{
namespace tensor
{

/**
 * \brief Debug string for SYCL kernel.
 *
 * Used in ``sycl::handler::parallel_for()``.
 *
 * \tparam T    Template parameter (inherited)
 */
template <typename T> class dpex_interval_step_kernel;

/**
 * \brief Debug string for SYCL kernel.
 *
 * Used in ``sycl::handler::parallel_for()``.
 *
 * \tparam T    Template parameter (inherited)
 * \tparam wT   Template parameter (inherited)
 */
template <typename T, typename wT> class dpex_affine_interval_kernel;

/**
 * \brief Indexing operator class to be used by an interval step.
 *
 * This will be used in the ``sycl::handler::parallel_for()``.
 *
 * \tparam T    Template parameter to specialize.
 */
template <typename T> class IntervalStepFunctor
{
private:
    T *p = nullptr; // Pointer to the tensor array.
    T start_v;      // Start of the interval.
    T step_v;       // Step of the interval.

public:
    /**
     * \brief Construct a new Interval Step Functor object
     *
     * \param dst_p     Pointer to the destination array.
     * \param v0        The start of the interval.
     * \param dv        The step of the interval.
     */
    IntervalStepFunctor(char *dst_p, T v0, T dv)
        : p(reinterpret_cast<T *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    /**
     * \brief   The indexing operator.
     *
     * At index ``wiid``, the operator computes a value and assign the value
     * into the array.
     *
     * \param wiid  The index value.
     */
    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        if constexpr (dpexrt_tensor::typeutils::is_complex<T>::value) {
            p[i] = T{start_v.real() + i * step_v.real(),
                     start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + (i * step_v);
        }
    }
};

/**
 * \brief Indexing operator class to be used by an affine interval.
 *
 * \tparam T    The template parameter to specialize.
 * \tparam wT   The template parameter for ``sycl::fma()``.
 */
template <typename T, typename wT> class AffineIntervalFunctor
{
private:
    T *p = nullptr; // Pointer to the tensor array.
    T start_v;      // The start of the interval.
    T end_v;        // The end of the interval.
    size_t n;       // The size of the array.

public:
    /**
     * \brief Construct a new Affine Interval Functor object
     *
     * \param dst_p     // Pointer to the destination array.
     * \param v0        // The start of the interval.
     * \param v1        // The end of the interval.
     * \param den       // The denominator value for the ``sycl::fma()``.
     */
    AffineIntervalFunctor(char *dst_p, T v0, T v1, size_t den)
        : p(reinterpret_cast<T *>(dst_p)), start_v(v0), end_v(v1),
          n((den == 0) ? 1 : den)
    {
    }

    /**
     * \brief   The indexing operator.
     *
     * At index ``wiid``, the operator computes a value and assign the value
     * into the array.
     *
     * \param wiid  The index value.
     */
    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        wT wc = wT(i) / n;
        wT w = wT(n - i) / n;
        if constexpr (dpexrt_tensor::typeutils::is_complex<T>::value) {
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
            p[i] = dpexrt_tensor::typeutils::convert_impl<
                T, decltype(affine_comb)>(affine_comb);
        }
    }
};

/**
 * \brief The interval step kernel.
 *
 * This function uses the ``IntervalStepFunctor<T>()`` to index the array
 * to assign value to that location. This runs a SYCL parallel loop.
 *
 * \tparam T            The type parameter for specialization.
 * \param exec_q        The ``sycl::queue`` to be used by the kernel.
 * \param nelems        The number of elements in the array, i.e. size.
 * \param start_v       The start of the interval.
 * \param step_v        The step of the interval.
 * \param array_data    The pointer to the array.
 * \param depends       A vector of events passed from the caller.
 * \return sycl::event  Return event to indicate termination.
 */
template <typename T>
sycl::event interval_step_kernel(sycl::queue exec_q,
                                 size_t nelems,
                                 T start_v,
                                 T step_v,
                                 char *array_data,
                                 const std::vector<sycl::event> &depends)
{
    dpexrt_tensor::typeutils::validate_type_for_device<T>(exec_q);

    sycl::event seq_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<dpex_interval_step_kernel<T>>(
            sycl::range<1>{nelems},
            IntervalStepFunctor<T>(array_data, start_v, step_v));
    });

    return seq_step_event;
}

/**
 * \brief   The affine interval kernel.
 *
 * This function uses the ``AffineIntervalFunctor<T>()`` to index the array
 * to assign value to that location. This runs a SYCL parallel loop.
 *
 * \tparam T                The type parameter for specialization.
 * \param exec_q            The ``sycl::queue`` to be used by the kernel.
 * \param nelems            The number of elements in the array, i.e. size.
 * \param start_v           The start of the interval.
 * \param end_v             The end of the interval.
 * \param include_endpoint  The flag to include the end point.
 * \param array_data        The pointer to the array.
 * \param depends           A vector of events passed from the caller.
 * \return sycl::event      Return event to indicate termination.
 */
template <typename T>
sycl::event affine_interval_kernel(sycl::queue &exec_q,
                                   size_t nelems,
                                   T start_v,
                                   T end_v,
                                   bool include_endpoint,
                                   char *array_data,
                                   const std::vector<sycl::event> &depends)
{
    dpexrt_tensor::typeutils::validate_type_for_device<T>(exec_q);
    bool device_supports_doubles = exec_q.get_device().has(sycl::aspect::fp64);
    sycl::event affine_seq_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (device_supports_doubles) {
            cgh.parallel_for<dpex_affine_interval_kernel<T, double>>(
                sycl::range<1>{nelems},
                AffineIntervalFunctor<T, double>(array_data, start_v, end_v,
                                                 (include_endpoint) ? nelems - 1
                                                                    : nelems));
        }
        else {
            cgh.parallel_for<dpex_affine_interval_kernel<T, float>>(
                sycl::range<1>{nelems},
                AffineIntervalFunctor<T, float>(array_data, start_v, end_v,
                                                (include_endpoint) ? nelems - 1
                                                                   : nelems));
        }
    });

    return affine_seq_step_event;
}

/**
 * \brief Function to call specialized ``interval_step_kernel<T>()``
 *
 * This function will specialize the kernel ``interval_step_kernel()``
 * and call it.
 *
 * \tparam T            The type parameter for specialization.
 * \param exec_q        The ``sycl::queue`` to be used by the kernel.
 * \param nelems        The number of elements in the array, i.e. size.
 * \param start         The start of the interval, opaque pointer.
 * \param step          The step of the interval, opaque pointer.
 * \param array_data    The pointer to the array.
 * \param depends       A vector of events passed from the caller.
 * \return sycl::event  Return event to indicate termination.
 */
template <typename T>
sycl::event interval_step(sycl::queue &exec_q,
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

    auto interval_step_event = interval_step_kernel<T>(
        exec_q, nelems, *start_v, *step_v, array_data, depends);

    return interval_step_event;
}

/**
 * \brief   Function to call specialized ``affine_interval_kernel<T>()``
 *
 * This function will specialize the kernel ``affine_interval_kernel()``
 * and call it.
 *
 * \tparam T                The type parameter for specialization.
 * \param exec_q            The ``sycl::queue`` to be used by the kernel.
 * \param nelems            The number of elements in the array, i.e. size.
 * \param start             The start of the interval, opaque pointer.
 * \param end               The end of the interval, opaque pointer.
 * \param include_endpoint  The flag to include the end point.
 * \param array_data        The pointer to the array.
 * \param depends           A vector of events passed from the caller.
 * \return sycl::event      Return event to indicate termination.
 */
template <typename T>
sycl::event affine_interval(sycl::queue &exec_q,
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

    auto affine_interval_step_event =
        affine_interval_kernel<T>(exec_q, nelems, *start_v, *end_v,
                                  include_endpoint, array_data, depends);

    return affine_interval_step_event;
}

/**
 * \brief Returns a function pointer to ``interval_step<T>()``.
 *
 * \tparam fnT  The function pointer type.
 * \tparam T    The type of the returned function to be specialized.
 */
template <typename fnT, typename T> struct IntervalStepFactory
{
    fnT get()
    {
        fnT f = interval_step<T>;
        return f;
    }
};

/**
 * \brief Returns a function pointer to ``affine_interval<T>()``.
 *
 * \tparam fnT  The function pointer type.
 * \tparam T    The type of the returned function to be specialized.
 */
template <typename fnT, typename T> struct AffineIntervalFactory
{
    fnT get()
    {
        fnT f = affine_interval<T>;
        return f;
    }
};

/**
 * \brief A function pointer to ``interval_step<T>()``
 */
typedef sycl::event (*interval_step_ptr_t)(sycl::queue &,
                                           size_t, // num_elements
                                           void *, // start_v
                                           void *, // end_v
                                           char *, // dst_data_ptr
                                           const std::vector<sycl::event> &);

/**
 * \brief A function pointer to ``affine_interval<T>()``
 */
typedef sycl::event (*affine_interval_ptr_t)(sycl::queue &,
                                             size_t, // num_elements
                                             void *, // start_v
                                             void *, // end_v
                                             bool,   // include_endpoint
                                             char *, // dst_data_ptr
                                             const std::vector<sycl::event> &);
} // namespace tensor
} // namespace kernel
} // namespace rt
} // namespace dpex

#endif
