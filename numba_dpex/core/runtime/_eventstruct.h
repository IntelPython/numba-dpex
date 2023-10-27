// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the numba-dpex native representation for a dpctl.SyclEvent
///
//===----------------------------------------------------------------------===//

#pragma once

#include "numba/core/runtime/nrt_external.h"

typedef struct
{
    NRT_MemInfo *meminfo;
    void *event_ref;
} eventstruct_t;