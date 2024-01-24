// SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "nrt_reserve_meminfo.h"

#include "_dbg_printer.h"
#include "syclinterface/dpctl_sycl_type_casters.hpp"
#include <CL/sycl.hpp>

extern "C"
{
    DPCTLSyclEventRef
    DPEXRT_nrt_acquire_meminfo_and_schedule_release(NRT_api_functions *nrt,
                                                    DPCTLSyclQueueRef QRef,
                                                    NRT_MemInfo **meminfo_array,
                                                    size_t meminfo_array_size,
                                                    DPCTLSyclEventRef *depERefs,
                                                    size_t nDepERefs,
                                                    int *status)
    {
        DPEXRT_DEBUG(drt_debug_print(
                         "DPEXRT-DEBUG: scheduling nrt meminfo release.\n"););

        using dpctl::syclinterface::unwrap;
        using dpctl::syclinterface::wrap;

        sycl::queue *q = unwrap<sycl::queue>(QRef);

        std::vector<NRT_MemInfo *> meminfo_vec(
            meminfo_array, meminfo_array + meminfo_array_size);

        for (size_t i = 0; i < meminfo_array_size; ++i) {
            nrt->acquire(meminfo_vec[i]);
        }

        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: acquired meminfo.\n"););

        try {
            sycl::event ht_ev = q->submit([&](sycl::handler &cgh) {
                for (size_t ev_id = 0; ev_id < nDepERefs; ++ev_id) {
                    cgh.depends_on(*(unwrap<sycl::event>(depERefs[ev_id])));
                }
                cgh.host_task([meminfo_array_size, meminfo_vec, nrt]() {
                    for (size_t i = 0; i < meminfo_array_size; ++i) {
                        nrt->release(meminfo_vec[i]);
                        DPEXRT_DEBUG(
                            drt_debug_print("DPEXRT-DEBUG: released meminfo "
                                            "from host_task.\n"););
                    }
                });
            });

            constexpr int result_ok = 0;

            *status = result_ok;
            auto e_ptr = new sycl::event(ht_ev);
            return wrap<sycl::event>(e_ptr);
        } catch (const std::exception &e) {
            constexpr int result_std_exception = 1;

            *status = result_std_exception;
            return nullptr;
        }

        constexpr int result_other_abnormal = 2;

        *status = result_other_abnormal;
        return nullptr;
    }
}
