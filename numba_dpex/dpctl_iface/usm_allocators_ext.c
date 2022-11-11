// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// A Python extension that defines an external allocator for Numba. The
/// new external allocator uses SYCL's USM shared allocator exposed by DPCTL's
/// C API (libDPCTLSyclInterface). The extension module is used by the
/// numpy_usm_shared module.
///
//===----------------------------------------------------------------------===//

#include "assert.h"
#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt.h"
#include "numba/core/runtime/nrt_external.h"
#include <stdbool.h>
#include <stdio.h>

// clang-format off
#if defined __has_include
    #if __has_include(<dpctl_sycl_interface.h>)
        #include <dpctl_sycl_interface.h>
    #else
        #include <dpctl_sycl_queue_interface.h>
        #include <dpctl_sycl_queue_manager.h>
        #include <dpctl_sycl_usm_interface.h>
    #endif
#else
    #include <dpctl_sycl_queue_interface.h>
    #include <dpctl_sycl_queue_manager.h>
    #include <dpctl_sycl_usm_interface.h>
#endif
// clang-format on

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

void *usmarray_get_ext_allocator(void) { return (void *)&usmarray_allocator; }

static PyObject *get_external_allocator(PyObject *self, PyObject *args)
{
    return PyLong_FromVoidPtr(usmarray_get_ext_allocator());
}

/*
 * Internal structure used for allocation and deallocation.
 * USM types:
 *   0 - shared
 *   1 - device
 *   2 - host
 */
typedef struct
{
    int usm_type;
    void *queue;
} AllocatorImpl;

/*
 * Allocate USM memory.
 */
static void *allocate(size_t size, void *opaque_data)
{
    AllocatorImpl *impl = (AllocatorImpl *)opaque_data;
    void *data = 0;

    if (impl->usm_type == 0) {
        data = (void *)DPCTLmalloc_shared(size, impl->queue);
    }
    else if (impl->usm_type == 1) {
        data = (void *)DPCTLmalloc_host(size, impl->queue);
    }
    else if (impl->usm_type == 2) {
        data = (void *)DPCTLmalloc_device(size, impl->queue);
    }

    return data;
}

/*
 * Deallocate USM memory.
 */
static void deallocate(void *data, void *opaque_data)
{
    AllocatorImpl *impl = (AllocatorImpl *)opaque_data;

    DPCTLfree_with_queue(data, impl->queue);
}

/*
 * Create external allocator.
 * NOTE: experimental. Could be deleted.
 */
static NRT_ExternalAllocator *create_allocator(int usm_type)
{
    AllocatorImpl *impl = malloc(sizeof(AllocatorImpl));
    impl->usm_type = usm_type;
    impl->queue = (void *)DPCTLQueueMgr_GetCurrentQueue();

    NRT_ExternalAllocator *allocator = malloc(sizeof(NRT_ExternalAllocator));
    allocator->malloc = allocate;
    allocator->realloc = NULL;
    allocator->free = deallocate;
    allocator->opaque_data = impl;

    return allocator;
}

/*
 * Release external allocator.
 * NOTE: experimental. Could be deleted.
 */
static void release_allocator(NRT_ExternalAllocator *allocator)
{
    AllocatorImpl *impl = (AllocatorImpl *)allocator->opaque_data;
    DPCTLQueue_Delete(impl->queue);

    free(impl);
    free(allocator);
}

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

/*
 * Initialize MemInfo with data.
 * NOTE: copy from numba/core/runtime/nrt.c
 */
void NRT_MemInfo_init(NRT_MemInfo *mi,
                      void *data,
                      size_t size,
                      NRT_dtor_function dtor,
                      void *dtor_info,
                      NRT_ExternalAllocator *external_allocator)
{
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    mi->external_allocator = external_allocator;
    NRT_Debug(nrt_debug_print("NRT_MemInfo_init mi=%p external_allocator=%p\n",
                              mi, external_allocator));
}

/*
 * Allocate MemInfo and initialize.
 * NOTE: copy from numba/core/runtime/nrt.c
 */
NRT_MemInfo *NRT_MemInfo_new(void *data,
                             size_t size,
                             NRT_dtor_function dtor,
                             void *dtor_info)
{
    NRT_MemInfo *mi = malloc(sizeof(NRT_MemInfo));
    NRT_Debug(nrt_debug_print("NRT_MemInfo_new mi=%p\n", mi));
    NRT_MemInfo_init(mi, data, size, dtor, dtor_info, NULL);
    return mi;
}

/*
 * Destructor for allocated USM memory.
 */
static void dtor(void *ptr, size_t size, void *info)
{
    AllocatorImpl *impl = (AllocatorImpl *)info;

    DPCTLfree_with_queue(ptr, impl->queue);
    DPCTLQueue_Delete(impl->queue);
    free(impl);
}

/*
 * Debugging printf function used internally
 * NOTE: copy from numba/core/runtime/nrt.c
 */
void nrt_debug_print(char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*
 * Create MemInfo with allocated USM memory with given USM type and queue.
 * USM types:
 *   0 - shared
 *   1 - device
 *   2 - host
 */
static NRT_MemInfo *DPRT_MemInfo_new(size_t size, int usm_type, void *queue)
{
    NRT_Debug(nrt_debug_print("DPRT_MemInfo_new size=%d usm_type=%d queue=%p\n",
                              size, usm_type, queue));

    AllocatorImpl *impl = malloc(sizeof(AllocatorImpl));
    impl->usm_type = usm_type;
    impl->queue = queue;

    void *data = allocate(size, impl);

    return NRT_MemInfo_new(data, size, dtor, impl);
}

/*
 * Helper function for creating default queue
 */
void *create_queue()
{
    DPCTLSyclQueueRef queue = DPCTLQueueMgr_GetCurrentQueue();

    NRT_Debug(nrt_debug_print("create_queue queue=%p\n", queue));

    return queue;
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

static PyObject *build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value)                                              \
    do {                                                                       \
        PyObject *o = PyLong_FromVoidPtr(value);                               \
        if (o == NULL)                                                         \
            goto error;                                                        \
        if (PyDict_SetItemString(dct, name, o)) {                              \
            Py_DECREF(o);                                                      \
            goto error;                                                        \
        }                                                                      \
        Py_DECREF(o);                                                          \
    } while (0)

    _declpointer("usmarray_get_ext_allocator", &usmarray_get_ext_allocator);
    _declpointer("create_allocator", &create_allocator);
    _declpointer("release_allocator", &release_allocator);
    _declpointer("DPRT_MemInfo_new", &DPRT_MemInfo_new);
    _declpointer("create_queue", &create_queue);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_usm_allocators_ext)
{
    PyObject *m;
    MOD_DEF(m, "numba_dpex._usm_allocators_ext", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    usmarray_memsys_init();
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
