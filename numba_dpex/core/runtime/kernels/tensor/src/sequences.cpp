// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <stdexcept>

#include <numpy/npy_common.h>

#include "../include/sequences.hpp"
#include "../include/dispatch.hpp"
#include "../include/typeutils.hpp"
#include "../include/api.h"

namespace dpexrt_tensor = dpex::rt::kernel::tensor;

static dpexrt_tensor::sequence_step_ptr_t
    sequence_step_dispatch_vector[dpexrt_tensor::typeutils::num_types];

static dpexrt_tensor::affine_sequence_ptr_t
    affine_sequence_dispatch_vector[dpexrt_tensor::typeutils::num_types];

extern "C" void NUMBA_DPEX_SYCL_KERNEL_init_sequence_step_dispatch_vectors()
{
    dpexrt_tensor::dispatch::DispatchVectorBuilder<
        dpexrt_tensor::sequence_step_ptr_t, dpexrt_tensor::SequenceStepFactory,
        dpexrt_tensor::typeutils::num_types>
        dvb;
    dvb.populate_dispatch_vector(sequence_step_dispatch_vector);
}

extern "C" void NUMBA_DPEX_SYCL_KERNEL_init_affine_sequence_dispatch_vectors()
{
    dpexrt_tensor::dispatch::DispatchVectorBuilder<
        dpexrt_tensor::affine_sequence_ptr_t,
        dpexrt_tensor::AffineSequenceFactory,
        dpexrt_tensor::typeutils::num_types>
        dvb;
    dvb.populate_dispatch_vector(affine_sequence_dispatch_vector);
}

extern "C" uint NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence(
    void *start,
    void *dt,
    arystruct_t *dst,
    int ndim,
    u_int8_t is_c_contiguous,
    int dst_typeid,
    const DPCTLSyclQueueRef exec_q)
{
    std::cout << "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence:"
              << " start = "
              << dpexrt_tensor::typeutils::caste_using_typeid(start, dst_typeid)
              << std::endl;

    std::cout << "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence:"
              << " dt = "
              << dpexrt_tensor::typeutils::caste_using_typeid(dt, dst_typeid)
              << std::endl;

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
    std::cout << "NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence:"
              << " len = " << len << std::endl;

    char *dst_data = reinterpret_cast<char *>(dst->data);

    // int dst_typeid = 7; // 7 = int64_t, 10 = float, 11 = double
    auto fn = sequence_step_dispatch_vector[dst_typeid];

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

// uint dpexrt_tensor::tensor::populate_arystruct_affine_sequence(
//     void *start,
//     void *end,
//     arystruct_t *dst,
//     int include_endpoint,
//     int ndim,
//     int is_c_contiguous,
//     const DPCTLSyclQueueRef exec_q,
//     const DPCTLEventVectorRef depends)
// {
//     if (ndim != 1) {
//         throw std::logic_error(
//             "populate_arystruct_linseq(): array must be 1D.");
//     }
//     if (!is_c_contiguous) {
//         throw std::logic_error(
//             "populate_arystruct_linseq(): array must be c-contiguous.");
//     }
//     /**
//     auto array_types = td_ns::usm_ndarray_types();
//     int dst_typenum = dst.get_typenum();
//     int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

//     py::ssize_t len = dst.get_shape(0);
//     if (len == 0) {
//         // nothing to do
//         return std::make_pair(sycl::event{}, sycl::event{});
//     }

//     char *dst_data = dst.get_data();
//     sycl::event linspace_affine_event;

//     auto fn = lin_space_affine_dispatch_vector[dst_typeid];

//     linspace_affine_event = fn(exec_q, static_cast<size_t>(len), start, end,
//                                include_endpoint, dst_data, depends);

//     return std::make_pair(
//         keep_args_alive(exec_q, {dst}, {linspace_affine_event}),
//         linspace_affine_event);
//     */
//     return 0;
// }
