// SPDX-FileCopyrightText: 2023 - 2024 Intel Corporation
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
#if 1
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

/*
 * Debugging printf function used internally
 */
static inline void drt_debug_print(const char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}
