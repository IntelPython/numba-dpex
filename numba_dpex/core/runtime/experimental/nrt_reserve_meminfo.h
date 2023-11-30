// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dpctl style function(s) that interruct with nrt meminfo and sycl.
///
//===----------------------------------------------------------------------===//

#ifndef _EXPERIMENTAL_H_
#define _EXPERIMENTAL_H_

#include "dpctl_capi.h"
#include "numba/core/runtime/nrt_external.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
     * @brief Acquires meminfos and schedules a host task to release them.
     *
     * @param    nrt            NRT public API functions,
     * @param    QRef           Queue reference,
     * @param    meminfo_array  Array of meminfo pointers to perform actions on,
     * @param    meminfo_array_size Length of meminfo_array,
     * @param    depERefs       Array of dependant events for the host task,
     * @param    nDepERefs      Length of depERefs,
     * @param    status         Variable to write status to. Same style as
     * dpctl,
     * @return   {return}       Event reference to the host task.
     */
    DPCTLSyclEventRef
    DPEXRT_nrt_acquire_meminfo_and_schedule_release(NRT_api_functions *nrt,
                                                    DPCTLSyclQueueRef QRef,
                                                    NRT_MemInfo **meminfo_array,
                                                    size_t meminfo_array_size,
                                                    DPCTLSyclEventRef *depERefs,
                                                    size_t nDepERefs,
                                                    int *status);
#ifdef __cplusplus
}
#endif

#endif /* _EXPERIMENTAL_H_ */
