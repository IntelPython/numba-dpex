// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file intervals.cpp
/// \brief Implements intervals.hpp and api.h
///
/// This file contains APIs to facilitate different tensor construction and
/// manipulation routines. These functions will be almost always invoked from
/// ``numba``'s ``cgutils`` module. As a result, they are designed as C
/// functions.
///
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include <numpy/npy_common.h>

#include "../include/api.h"
#include "../include/dispatch.hpp"
#include "../include/intervals.hpp"
#include "../include/typeutils.hpp"

// Shorthand namespace for easy coding.
namespace dpexrt_tensor = dpex::rt::kernel::tensor;

/**
 * \brief The dispatch vector to contain specialized 14 ``interval_step<T>()``
 * functions.
 */
static dpexrt_tensor::interval_step_ptr_t
    interval_step_dispatch_vector[dpexrt_tensor::typeutils::num_types];

/**
 * \brief The dispatch vector to contain specialized 14 ``affine_interval<T>()``
 * functions.
 */
static dpexrt_tensor::affine_interval_ptr_t
    affine_interval_dispatch_vector[dpexrt_tensor::typeutils::num_types];

/**
 * \brief Dispatch vector initializer function for intervals.
 *
 * This function populates an array of function pointers where
 * each function is specialized from it's template. Each function
 * populates and USM array of values of a given type. This function
 * populates the array using a starting point and an increment step.
 */
extern "C" void NUMBA_DPEX_SYCL_KERNEL_init_interval_step_dispatch_vectors()
{
    dpexrt_tensor::dispatch::DispatchVectorBuilder<
        dpexrt_tensor::interval_step_ptr_t, dpexrt_tensor::IntervalStepFactory,
        dpexrt_tensor::typeutils::num_types>
        dvb;
    dvb.populate_dispatch_vector(interval_step_dispatch_vector);
}

/**
 * \brief Dispatch vector initializer function for affine intervals.
 *
 * This function populates an array of function pointers where
 * each function is specialized from it's template. Each function
 * populates and USM array of values of a given type. This function
 * populates the array using a starting point and an end point.
 */
extern "C" void NUMBA_DPEX_SYCL_KERNEL_init_affine_interval_dispatch_vectors()
{
    dpexrt_tensor::dispatch::DispatchVectorBuilder<
        dpexrt_tensor::affine_interval_ptr_t,
        dpexrt_tensor::AffineIntervalFactory,
        dpexrt_tensor::typeutils::num_types>
        dvb;
    dvb.populate_dispatch_vector(affine_interval_dispatch_vector);
}

/**
 * \brief Calls linear intervals dispatch function(s).
 *
 * This function calls a single instance of the array filling function
 * stored by NUMBA_DPEX_SYCL_KERNEL_init_interval_step_dispatch_vectors(),
 * where each templated function is specialized to a data type. Each
 * specialized function can be indexed with ``dst_typeid``, e.g. ``0`` for
 * ``bool``,
 * ``1`` for ``int8_t`` and so on.
 *
 * \param start             The start of the interval.
 * \param dt                The increment of the interval.
 * \param dst               The destination ``arystruct_t``.
 * \param ndim              The cardinality of the array.
 * \param is_c_contiguous   A flag indicating whether the array is
 *                              c-contiguous.
 * \param dst_typeid        The type index for the data type to which a
 *                              templated function will be specialized.
 * \param exec_q            The opaque pointer to ``sycl::queue``
 * \return unsigned int     Returns 0 on success, else 1 if fails.
 */
extern "C" unsigned int NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_interval(
    void *start,
    void *dt,
    arystruct_t *dst,
    int ndim,
    bool is_c_contiguous,
    int dst_typeid,
    const DPCTLSyclQueueRef exec_q)
{
    if (ndim != 1) {
        throw std::logic_error(
            "populate_arystruct_linseq(): array must be 1D.");
    }
    if (!is_c_contiguous) {
        throw std::logic_error(
            "populate_arystruct_linseq(): array must be c-contiguous.");
    }

    size_t len = static_cast<size_t>(dst->nitems);
    if (len == 0)
        return 0;

    char *dst_data = reinterpret_cast<char *>(dst->data);

    auto fn = interval_step_dispatch_vector[dst_typeid];
    sycl::queue *queue = reinterpret_cast<sycl::queue *>(exec_q);
    std::vector<sycl::event> depends = std::vector<sycl::event>();
    sycl::event linspace_step_event =
        fn(*queue, len, start, dt, dst_data, depends);

    linspace_step_event.wait_and_throw();

    if (linspace_step_event
            .get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete)
        return 0;
    else
        return 1;
}

/**
 * \brief Calls affine intervals dispatch function(s).
 *
 * This function calls a single instance of the array filling function
 * stored by NUMBA_DPEX_SYCL_KERNEL_init_affine_interval_dispatch_vectors(),
 * where each templated function is specialized to a data type. Each
 * specialized function can be indexed with ``dst_typeid``, e.g. ``0`` for
 * ``bool`` and ``1`` for ``int8_t`` and so on.
 *
 * \param start             The start of the interval.
 * \param end               The end of the interval.
 * \param dst               The destination ``arystruct_t``.
 * \param include_endpoint  A flag indicating whether to include ``end``
 *                              in the interval.
 * \param ndim              The cardinality of the array.
 * \param is_c_contiguous   A flag indicating whether the array is
 *                              c-contiguous.
 * \param dst_typeid        The type index for the data type to which a
 *                              templated function will be specialized.
 * \param exec_q            The opaque pointer to ``sycl::queue``
 * \return unsigned int     Returns 0 on success, else 1 if fails.
 */
extern "C" unsigned int
NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_affine_interval(
    void *start,
    void *end,
    arystruct_t *dst,
    bool include_endpoint,
    int ndim,
    bool is_c_contiguous,
    int dst_typeid,
    const DPCTLSyclQueueRef exec_q)
{
    if (ndim != 1) {
        throw std::logic_error(
            "populate_arystruct_linseq(): array must be 1D.");
    }
    if (!is_c_contiguous) {
        throw std::logic_error(
            "populate_arystruct_linseq(): array must be c-contiguous.");
    }

    size_t len = static_cast<size_t>(dst->nitems);
    if (len == 0)
        return 0;

    char *dst_data = reinterpret_cast<char *>(dst->data);
    sycl::queue *queue = reinterpret_cast<sycl::queue *>(exec_q);
    std::vector<sycl::event> depends = std::vector<sycl::event>();

    auto fn = affine_interval_dispatch_vector[dst_typeid];

    sycl::event linspace_affine_event =
        fn(*queue, len, start, end, include_endpoint, dst_data, depends);

    linspace_affine_event.wait_and_throw();

    if (linspace_affine_event
            .get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete)
        return 0;
    else
        return 1;
}
