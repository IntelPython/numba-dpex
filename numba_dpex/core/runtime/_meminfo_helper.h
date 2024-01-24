// SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _NRT_ARRAY_STRUCT_H_
#define _NRT_ARRAY_STRUCT_H_
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <numpy/ndarrayobject.h>

#include "numba/_numba_common.h"
#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt.h"

/*
 * The MemInfo structure.
 * NOTE: copy from numba/core/runtime/nrt.c
 */
struct MemInfo
{
    size_t refct;
    NRT_dtor_function dtor;
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
    NRT_ExternalAllocator *external_allocator;
};

/*!
 * @brief A wrapper struct to store a MemInfo pointer along with the PyObject
 * that is associated with the MeMinfo.
 *
 * The struct is stored in the dtor_info attribute of a MemInfo object and
 * used by the destructor to free the MemInfo and DecRef the Pyobject.
 *
 */
typedef struct
{
    PyObject *owner;
    NRT_MemInfo *mi;
} MemInfoDtorInfo;

typedef struct
{
    PyObject_HEAD NRT_MemInfo *meminfo;
} MemInfoObject;

#endif /* _NRT_ARRAY_STRUCT_H_ */
