// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dpctl.hpp"
#include <CL/sycl.hpp>

namespace std
{

size_t
hash<DPCTLSyclDeviceRef>::operator()(const DPCTLSyclDeviceRef &DRef) const
{
    using dpctl::syclinterface::unwrap;
    return hash<sycl::device>()(*unwrap<sycl::device>(DRef));
}

size_t
hash<DPCTLSyclContextRef>::operator()(const DPCTLSyclContextRef &CRef) const
{
    using dpctl::syclinterface::unwrap;
    return hash<sycl::context>()(*unwrap<sycl::context>(CRef));
}
} // namespace std
