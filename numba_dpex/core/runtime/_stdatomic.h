// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Define a small subset of stdatomic.h to compile on Windows.
///
//===----------------------------------------------------------------------===//

#ifndef COMPAT_ATOMICS_WIN32_STDATOMIC_H
#define COMPAT_ATOMICS_WIN32_STDATOMIC_H

#define WIN32_LEAN_AND_MEAN
#include <stddef.h>
#include <stdint.h>
#include <windows.h>

#define ATOMIC_FLAG_INIT 0

#define ATOMIC_VAR_INIT(value) (value)

typedef intptr_t atomic_size_t;

#ifdef _WIN64
#define atomic_fetch_add(object, operand)                                      \
    InterlockedExchangeAdd64(object, operand)
#endif /* _WIN64 */

#define atomic_fetch_add_explicit(object, operand, order)                      \
    atomic_fetch_add(object, operand)

#endif /* COMPAT_ATOMICS_WIN32_STDATOMIC_H */
