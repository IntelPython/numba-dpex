// Copyright 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A Python extension that defines an external allocator for Numba. The
/// new external allocator uses SYCL's USM shared allocator exposed by DPCTL's
/// C API (libDPCTLSyclInterface). The extension module is used by the
/// numpy_usm_shared module.
///
//===----------------------------------------------------------------------===//

#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt_external.h"
#include "assert.h"
#include <stdio.h>
#include <stdbool.h>
#if defined __has_include
#  if __has_include(<dpctl_sycl_interface.h>)
#    include <dpctl_sycl_interface.h>
#  else
#    include <dpctl_sycl_queue_interface.h>
#    include <dpctl_sycl_queue_manager.h>
#    include <dpctl_sycl_usm_interface.h>
#  endif
#else
#  include <dpctl_sycl_queue_interface.h>
#  include <dpctl_sycl_queue_manager.h>
#  include <dpctl_sycl_usm_interface.h>
#endif

NRT_ExternalAllocator usmarray_allocator;
NRT_external_malloc_func internal_allocator = NULL;
NRT_external_free_func internal_free = NULL;
void *(*get_queue_internal)(void) = NULL;
void (*free_queue_internal)(void *) = NULL;

void *save_queue_allocator(size_t size, void *opaque)
{
    // Allocate a pointer-size more space than needed.
    size_t new_size = size + sizeof(void *);
    // Get a copy of the current queue
    void *cur_queue = (void *)DPCTLQueueMgr_GetCurrentQueue();
    // Use that queue to allocate.
    void *data = (void *)DPCTLmalloc_shared(new_size, cur_queue);
    // Set first pointer-sized data in allocated space to be the current queue.
    *(void **)data = cur_queue;
    // Return the pointer after this queue in memory.
    return (char *)data + sizeof(void *);
}

void save_queue_deallocator(void *data, void *opaque)
{
    // Compute original allocation location by subtracting the length
    // of the queue pointer from the data location that Numba thinks
    // starts the object.
    void *orig_data = (char *)data - sizeof(void *);
    // Get the queue from the original data by derefencing the first qword.
    void *obj_queue = *(void **)orig_data;
    // Free the space using the correct queue.
    DPCTLfree_with_queue(orig_data, obj_queue);
    // Free the queue itself.
    DPCTLQueue_Delete(obj_queue);
}

void usmarray_memsys_init(void)
{
    usmarray_allocator.malloc = save_queue_allocator;
    usmarray_allocator.realloc = NULL;
    usmarray_allocator.free = save_queue_deallocator;
    usmarray_allocator.opaque_data = NULL;
}

void *usmarray_get_ext_allocator(void)
{
    return (void *)&usmarray_allocator;
}

static PyObject *
get_external_allocator(PyObject *self, PyObject *args)
{
    return PyLong_FromVoidPtr(usmarray_get_ext_allocator());
}

static PyMethodDef ext_methods[] = {
// clang-format off
#define declmethod_noargs(func)                     \
    {                                               \
        #func, (PyCFunction)func, METH_NOARGS, NULL \
    }
    declmethod_noargs(get_external_allocator),
    {NULL},
#undef declmethod_noargs
};
// clang-format on

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value)                \
    do                                           \
    {                                            \
        PyObject *o = PyLong_FromVoidPtr(value); \
        if (o == NULL)                           \
            goto error;                          \
        if (PyDict_SetItemString(dct, name, o))  \
        {                                        \
            Py_DECREF(o);                        \
            goto error;                          \
        }                                        \
        Py_DECREF(o);                            \
    } while (0)

    _declpointer("usmarray_get_ext_allocator", &usmarray_get_ext_allocator);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_usm_shared_allocator_ext)
{
    PyObject *m;
    MOD_DEF(m, "numba_dppy._usm_shared_allocator_ext", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    usmarray_memsys_init();
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
