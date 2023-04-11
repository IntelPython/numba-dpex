// SPDX-FileCopyrightText: 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// A helper macro to print debug prints.
///
//===----------------------------------------------------------------------===//

#pragma once

/* Debugging facilities - enabled at compile-time */
/* #undef NDEBUG */
#if 0
#include <stdio.h>
#define DPEXRT_DEBUG(X)                                                        \
    {                                                                          \
        X;                                                                     \
        fflush(stdout);                                                        \
    }
#else
#define DPEXRT_DEBUG(X)                                                        \
    if (0) {                                                                   \
        X;                                                                     \
    }
#endif
