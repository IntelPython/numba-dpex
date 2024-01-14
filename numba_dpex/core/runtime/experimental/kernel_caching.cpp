// SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "kernel_caching.h"
#include <unordered_map>

extern "C"
{
#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#include "_dbg_printer.h"

#include "numba/core/runtime/nrt_external.h"
}

#include "syclinterface/dpctl_sycl_type_casters.hpp"
#include "tools/boost_hash.hpp"
#include "tools/dpctl.hpp"

using CacheKey = std::tuple<DPCTLSyclContextRef, DPCTLSyclDeviceRef, size_t>;

namespace std
{
template <> struct hash<CacheKey>
{
    size_t operator()(const CacheKey &ck) const
    {
        std::size_t seed = 0;
        boost::hash_combine(seed, std::get<DPCTLSyclDeviceRef>(ck));
        boost::hash_combine(seed, std::get<DPCTLSyclContextRef>(ck));
        boost::hash_detail::hash_combine_impl(seed, std::get<size_t>(ck));
        return seed;
    }
};
template <> struct equal_to<CacheKey>
{
    constexpr bool operator()(const CacheKey &lhs, const CacheKey &rhs) const
    {
        return DPCTLDevice_AreEq(std::get<DPCTLSyclDeviceRef>(lhs),
                                 std::get<DPCTLSyclDeviceRef>(rhs)) &&
               DPCTLContext_AreEq(std::get<DPCTLSyclContextRef>(lhs),
                                  std::get<DPCTLSyclContextRef>(rhs)) &&
               std::get<size_t>(lhs) == std::get<size_t>(rhs);
    }
};
} // namespace std

// TODO: add cache cleaning
// https://github.com/IntelPython/numba-dpex/issues/1240
std::unordered_map<CacheKey, DPCTLSyclKernelRef> sycl_kernel_cache =
    std::unordered_map<CacheKey, DPCTLSyclKernelRef>();

template <class M, class Key, class F>
typename M::mapped_type &get_else_compute(M &m, Key const &k, F f)
{
    typedef typename M::mapped_type V;
    std::pair<typename M::iterator, bool> r =
        m.insert(typename M::value_type(k, V()));
    V &v = r.first->second;
    if (r.second) {
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: building kernel.\n"););
        f(v);
    }
    else {
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: using cached kernel.\n"););
        DPCTLDevice_Delete(std::get<DPCTLSyclDeviceRef>(k));
        DPCTLContext_Delete(std::get<DPCTLSyclContextRef>(k));
    }
    return v;
}

extern "C"
{
    DPCTLSyclKernelRef DPEXRT_build_or_get_kernel(const DPCTLSyclContextRef ctx,
                                                  const DPCTLSyclDeviceRef dev,
                                                  size_t il_hash,
                                                  const char *il,
                                                  size_t il_length,
                                                  const char *compile_opts,
                                                  const char *kernel_name)
    {
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: in build or get kernel.\n"););

        CacheKey key = std::make_tuple(ctx, dev, il_hash);

        DPEXRT_DEBUG(auto ctx_hash = std::hash<DPCTLSyclContextRef>{}(ctx);
                     auto dev_hash = std::hash<DPCTLSyclDeviceRef>{}(dev);
                     drt_debug_print("DPEXRT-DEBUG: key hashes: %d %d %d.\n",
                                     ctx_hash, dev_hash, il_hash););

        auto k_ref = get_else_compute(
            sycl_kernel_cache, key,
            [ctx, dev, il, il_length, compile_opts,
             kernel_name](DPCTLSyclKernelRef &k_ref) {
                auto kb_ref = DPCTLKernelBundle_CreateFromSpirv(
                    ctx, dev, il, il_length, compile_opts);
                k_ref = DPCTLKernelBundle_GetKernel(kb_ref, kernel_name);
                DPCTLKernelBundle_Delete(kb_ref);
            });

        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: kernel hash size: %d.\n",
                                     sycl_kernel_cache.size()););

        return DPCTLKernel_Copy(k_ref);
    }

    size_t DPEXRT_kernel_cache_size() { return sycl_kernel_cache.size(); }
}
