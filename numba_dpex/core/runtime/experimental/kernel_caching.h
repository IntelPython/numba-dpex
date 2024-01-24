// SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dpex run time function(s) that cache kernel on device.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#ifdef __cplusplus
extern "C"
{
#endif
    /*!
     * @brief returns dpctl kernel reference for the SPIRV file on particular
     * device. Compiles only first time, all others will use cache for the same
     * input. It steals reference to context and device because we need to keep
     * it alive for cache keys.
     *
     * @param    ctx            Context reference,
     * @param    dev            Device reference,
     * @param    il_hash        Hash of the SPIRV binary data,
     * @param    il             SPIRV binary data,
     * @param    il_length      SPIRV binary data size,
     * @param    compile_opts   compile options,
     * @param    kernel_name    kernel name inside SPIRV binary data to return
     * reference to.
     *
     * @return   {return}       Kernel reference to the compiled SPIR-V.
     */
    DPCTLSyclKernelRef DPEXRT_build_or_get_kernel(const DPCTLSyclContextRef ctx,
                                                  const DPCTLSyclDeviceRef dev,
                                                  size_t il_hash,
                                                  const char *il,
                                                  size_t il_length,
                                                  const char *compile_opts,
                                                  const char *kernel_name);

    /*!
     * @brief returns cache size. Intended for test purposes only
     *
     * @return   {return}       Kernel cache size.
     */
    size_t DPEXRT_kernel_cache_size();
#ifdef __cplusplus
}
#endif
