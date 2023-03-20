// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
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

/* WARNING: Do not remove this, only modify it! It is a version guard to
 * act as a reminder to update this struct on Python version update! */
#if (PY_MAJOR_VERSION == 3)
#if !((PY_MINOR_VERSION == 8) || (PY_MINOR_VERSION == 9) ||                    \
      (PY_MINOR_VERSION == 10))
#error "Python minor version is not supported."
#endif
#else
#error "Python major version is not supported."
#endif
/* END WARNING*/

#endif /* _NRT_PYTHON_HELPER_H_ */
