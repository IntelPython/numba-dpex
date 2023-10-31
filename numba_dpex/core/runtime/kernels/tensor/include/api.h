// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <Python.h>
#include <numpy/npy_common.h>
#include <numba/_arraystruct.h>

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
    // Dispatch vector initializer functions.
    void NUMBA_DPEX_SYCL_KERNEL_init_sequence_step_dispatch_vectors();
    void NUMBA_DPEX_SYCL_KERNEL_init_affine_sequence_dispatch_vectors();

    // Call linear sequences dispatch functions.
    uint NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_sequence(
        void *start,
        void *dt,
        arystruct_t *dst,
        int ndim,
        u_int8_t is_c_contiguous,
        int dst_typeid,
        const DPCTLSyclQueueRef exec_q);

    // Call linear affine sequences dispatch functions.
    uint NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_affine_sequence(
        void *start,
        void *end,
        arystruct_t *dst,
        u_int8_t include_endpoint,
        int ndim,
        u_int8_t is_c_contiguous,
        const DPCTLSyclQueueRef exec_q);

#ifdef __cplusplus
}
#endif
