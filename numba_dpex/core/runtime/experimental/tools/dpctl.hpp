// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines overloads to dpctl library that eventually must be ported there.
///
//===----------------------------------------------------------------------===//

#pragma once
#include "syclinterface/dpctl_sycl_type_casters.hpp"

namespace std
{
template <> struct hash<DPCTLSyclDeviceRef>
{
    size_t operator()(const DPCTLSyclDeviceRef &DRef) const;
};

template <> struct hash<DPCTLSyclContextRef>
{
    size_t operator()(const DPCTLSyclContextRef &CRef) const;
};
} // namespace std
