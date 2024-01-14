// SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the numba-dpex native representation for a dpctl.tensor.usm_ndarray
///
//===----------------------------------------------------------------------===//

#pragma once

#include <Python.h>
#include <numpy/npy_common.h>

typedef struct
{
    void *meminfo;
    PyObject *parent;
    npy_intp nitems;
    npy_intp itemsize;
    void *data;
    void *sycl_queue;

    npy_intp shape_and_strides[];
} usmarystruct_t;
