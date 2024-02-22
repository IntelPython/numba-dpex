// SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dpctl style function(s) that interact with nrt meminfo and sycl.
///
//===----------------------------------------------------------------------===//

#ifndef _DPCTL_SUBMIT_RANGE_H_
#define _DPCTL_SUBMIT_RANGE_H_

#include "dpctl_capi.h"
#include "numba/core/runtime/nrt_external.h"

#ifdef __cplusplus
extern "C"
{
#endif

    __dpctl_give DPCTLSyclEventRef
    DpexDPCTLQueue_SubmitRange(__dpctl_keep const DPCTLSyclKernelRef KRef,
                               __dpctl_keep const DPCTLSyclQueueRef QRef,
                               __dpctl_keep void **Args,
                               __dpctl_keep const DPCTLKernelArgType *ArgTypes,
                               size_t NArgs,
                               __dpctl_keep const size_t Range[3],
                               size_t NDims,
                               __dpctl_keep const DPCTLSyclEventRef *DepEvents,
                               size_t NDepEvents);

    __dpctl_give DPCTLSyclEventRef DpexDPCTLQueue_SubmitNDRange(
        __dpctl_keep const DPCTLSyclKernelRef KRef,
        __dpctl_keep const DPCTLSyclQueueRef QRef,
        __dpctl_keep void **Args,
        __dpctl_keep const DPCTLKernelArgType *ArgTypes,
        size_t NArgs,
        __dpctl_keep const size_t gRange[3],
        __dpctl_keep const size_t lRange[3],
        size_t NDims,
        __dpctl_keep const DPCTLSyclEventRef *DepEvents,
        size_t NDepEvents);

#ifdef __cplusplus
}
#endif

#endif /* _DPCTL_SUBMIT_RANGE_H_ */
