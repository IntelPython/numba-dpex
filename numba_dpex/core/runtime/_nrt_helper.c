// SPDX-FileCopyrightText: 2020 - 2024 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "_nrt_helper.h"
#include <assert.h>

#include <stdbool.h>

#ifdef _MSC_VER
#include <stddef.h>
#include <stdint.h>
#include <windows.h>
typedef intptr_t atomic_size_t;
#ifdef _WIN64
#define atomic_fetch_add(obj, op) InterlockedExchangeAdd64(obj, op)
#endif
#define atomic_fetch_add_explicit(obj, op, od) atomic_fetch_add(obj, op)
#else
#include <stdatomic.h>
#endif

#include "_dbg_printer.h"

/*
 * Global resources.
 */
struct NRT_MemSys
{
    /* Shutdown flag */
    int shutting;
    /* Stats */
    struct
    {
        bool enabled;
        atomic_size_t alloc;
        atomic_size_t free;
        atomic_size_t mi_alloc;
        atomic_size_t mi_free;
    } stats;
    /* System allocation functions */
    struct
    {
        NRT_malloc_func malloc;
        NRT_realloc_func realloc;
        NRT_free_func free;
    } allocator;
};

/* The Memory System object */
static struct NRT_MemSys TheMSys;

// following funcs are copied from numba/core/runtime/nrt.cpp
void *NRT_MemInfo_external_allocator(NRT_MemInfo *mi)
{
    NRT_Debug(drt_debug_print(
        "NRT_MemInfo_external_allocator meminfo: %p external_allocator: %p\n",
        mi, mi->external_allocator));
    return mi->external_allocator;
}

void *NRT_MemInfo_data(NRT_MemInfo *mi) { return mi->data; }

void NRT_MemInfo_release(NRT_MemInfo *mi)
{
    assert(mi->refct > 0 && "RefCt cannot be 0");
    /* RefCt drop to zero */
    if ((--(mi->refct)) == 0) {
        NRT_MemInfo_call_dtor(mi);
    }
}

void NRT_MemInfo_call_dtor(NRT_MemInfo *mi)
{
    NRT_Debug(drt_debug_print("NRT_MemInfo_call_dtor %p\n", mi));
    if (mi->dtor && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

void NRT_MemInfo_acquire(NRT_MemInfo *mi)
{
    // NRT_Debug(drt_debug_print("NRT_MemInfo_acquire %p refct=%zu\n", mi,
    // mi->refct.load()));
    assert(mi->refct > 0 && "RefCt cannot be zero");
    mi->refct++;
}

size_t NRT_MemInfo_size(NRT_MemInfo *mi) { return mi->size; }

void *NRT_MemInfo_parent(NRT_MemInfo *mi) { return mi->dtor_info; }

size_t NRT_MemInfo_refcount(NRT_MemInfo *mi)
{
    /* Should never returns 0 for a valid MemInfo */
    if (mi && mi->data)
        return mi->refct;
    else {
        return (size_t)-1;
    }
}

void NRT_Free(void *ptr)
{
    NRT_Debug(drt_debug_print("NRT_Free %p\n", ptr));
    TheMSys.allocator.free(ptr);
    if (TheMSys.stats.enabled) {
        TheMSys.stats.free++;
    }
}

void NRT_dealloc(NRT_MemInfo *mi)
{
    NRT_Debug(
        drt_debug_print("NRT_dealloc meminfo: %p external_allocator: %p\n", mi,
                        mi->external_allocator));
    if (mi->external_allocator) {
        mi->external_allocator->free(mi, mi->external_allocator->opaque_data);
        if (TheMSys.stats.enabled) {
            TheMSys.stats.free++;
        }
    }
    else {
        NRT_Free(mi);
    }
}

void NRT_MemInfo_destroy(NRT_MemInfo *mi)
{
    NRT_dealloc(mi);
    if (TheMSys.stats.enabled) {
        TheMSys.stats.mi_free++;
    }
}

void NRT_MemInfo_pyobject_dtor(void *data)
{
    PyGILState_STATE gstate;
    PyObject *ownerobj = data;

    gstate = PyGILState_Ensure(); /* ensure the GIL */
    Py_DECREF(data);              /* release the python object */
    PyGILState_Release(gstate);   /* release the GIL */

    DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: pyobject destructor\n"););
}
