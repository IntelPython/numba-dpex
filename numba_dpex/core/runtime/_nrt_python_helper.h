// SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// Re-definition of NRT functions for marshalling from / to Python objects
/// defined in numba/core/runtime/_nrt_python.c.
///
//===----------------------------------------------------------------------===//

#ifndef _NRT_PYTHON_HELPER_H_
#define _NRT_PYTHON_HELPER_H_

#define NO_IMPORT_ARRAY
#include "_meminfo_helper.h"

/*!
 * @brief A pyTypeObject to describe a Python object to wrap Numba's MemInfo
 *
 */
extern PyTypeObject MemInfoType;

void MemInfo_dealloc(MemInfoObject *self);
int MemInfo_init(MemInfoObject *self, PyObject *args, PyObject *kwds);
int MemInfo_getbuffer(PyObject *exporter, Py_buffer *view, int flags);
PyObject *MemInfo_acquire(MemInfoObject *self);
PyObject *MemInfo_release(MemInfoObject *self);
PyObject *MemInfo_get_data(MemInfoObject *self, void *closure);
PyObject *MemInfo_get_refcount(MemInfoObject *self, void *closure);
PyObject *MemInfo_get_external_allocator(MemInfoObject *self, void *closure);
PyObject *MemInfo_get_parent(MemInfoObject *self, void *closure);

#endif /* _NRT_PYTHON_HELPER_H_ */
