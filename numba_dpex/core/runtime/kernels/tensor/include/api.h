// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file api.h
/// \brief C APIs to provide functionalities of tensor constructors.
///
/// This file contains APIs to facilitate different tensor construction and
/// manipulation routines. These functions will be almost always invoked from
/// ``numba``'s ``cgutils`` module. As a result, they are designed as C
/// functions.
///
//===----------------------------------------------------------------------===//

#ifndef __API_H__
#define __API_H__

#include <Python.h>
#include <numba/_arraystruct.h>
#include <numpy/npy_common.h>
#include <stdint.h>

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
    /**
     * \brief Dispatch vector initializer function for intervals.
     *
     * This function populates an array of function pointers where
     * each function is specialized from it's template. Each function
     * populates and USM array of values of a given type. This function
     * populates the array using a starting point and an increment step.
     */
    void NUMBA_DPEX_SYCL_KERNEL_init_interval_step_dispatch_vectors();

    /**
     * \brief Dispatch vector initializer function for affine intervals.
     *
     * This function populates an array of function pointers where
     * each function is specialized from it's template. Each function
     * populates and USM array of values of a given type. This function
     * populates the array using a starting point and an end point.
     */
    void NUMBA_DPEX_SYCL_KERNEL_init_affine_interval_dispatch_vectors();

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
    unsigned int NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_interval(
        void *start,
        void *dt,
        arystruct_t *dst,
        int ndim,
        bool is_c_contiguous,
        int dst_typeid,
        const DPCTLSyclQueueRef exec_q);

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
    unsigned int NUMBA_DPEX_SYCL_KERNEL_populate_arystruct_affine_interval(
        void *start,
        void *end,
        arystruct_t *dst,
        bool include_endpoint,
        int ndim,
        bool is_c_contiguous,
        int dst_typeid,
        const DPCTLSyclQueueRef exec_q);

#ifdef __cplusplus
}
#endif
#endif // __API_H__
