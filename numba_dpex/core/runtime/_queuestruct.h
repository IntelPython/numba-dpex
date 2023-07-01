// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the numba-dpex native representation for a dpctl.SyclQueue
///
//===----------------------------------------------------------------------===//

#pragma once

#include <Python.h>

typedef struct
{
    PyObject *parent;
    void *queue_ref;
} queuestruct_t;
