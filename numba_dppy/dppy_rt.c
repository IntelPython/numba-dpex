#include "numba/_pymodule.h"
#include "numba/core/runtime/nrt_external.h"
#include "assert.h"
#include <stdio.h>
#if !defined _WIN32
   #include <dlfcn.h>
#else
   #include <windows.h>
#endif

NRT_ExternalAllocator usmarray_allocator;
NRT_external_malloc_func internal_allocator = NULL;
NRT_external_free_func internal_free = NULL;
void *(*get_queue_internal)(void) = NULL;
void (*free_queue_internal)(void*) = NULL;

void * save_queue_allocator(size_t size, void *opaque) {
    // Allocate a pointer-size more space than neded.
    int new_size = size + sizeof(void*);
    // Get the current queue
    void *cur_queue = get_queue_internal(); // this makes a copy
    // Use that queue to allocate.
    void *data = internal_allocator(new_size, cur_queue);
    // Set first pointer-sized data in allocated space to be the current queue.
    *(void**)data = cur_queue;
    // Return the pointer after this queue in memory.
    return (char*)data + sizeof(void*);
}

void save_queue_deallocator(void *data, void *opaque) {
    // Compute original allocation location by subtracting the length
    // of the queue pointer from the data location that Numba thinks
    // starts the object.
    void *orig_data = (char*)data - sizeof(void*);
    // Get the queue from the original data by derefencing the first qword.
    void *obj_queue = *(void**)orig_data;
    // Free the space using the correct queue.
    internal_free(orig_data, obj_queue);
    // Free the queue itself.
    free_queue_internal(obj_queue);
}

void usmarray_memsys_init(void) {
    #if !defined _WIN32
        char *lib_name = "libDPCTLSyclInterface.so";
        char *malloc_name = "DPCTLmalloc_shared";
        char *free_name = "DPCTLfree_with_queue";
        char *get_queue_name = "DPCTLQueueMgr_GetCurrentQueue";
        char *free_queue_name = "DPCTLQueue_Delete";

        void *sycldl = dlopen(lib_name, RTLD_NOW);
        assert(sycldl != NULL);
        internal_allocator = (NRT_external_malloc_func)dlsym(sycldl, malloc_name);
        usmarray_allocator.malloc = save_queue_allocator;
        if (internal_allocator == NULL) {
            printf("Did not find %s in %s\n", malloc_name, lib_name);
            exit(-1);
        }

        usmarray_allocator.realloc = NULL;

        internal_free = (NRT_external_free_func)dlsym(sycldl, free_name);
        usmarray_allocator.free = save_queue_deallocator;
        if (internal_free == NULL) {
            printf("Did not find %s in %s\n", free_name, lib_name);
            exit(-1);
        }

        get_queue_internal = (void *(*)(void))dlsym(sycldl, get_queue_name);
        if (get_queue_internal == NULL) {
            printf("Did not find %s in %s\n", get_queue_name, lib_name);
            exit(-1);
        }
        usmarray_allocator.opaque_data = NULL;

        free_queue_internal = (void (*)(void*))dlsym(sycldl, free_queue_name);
        if (free_queue_internal == NULL) {
            printf("Did not find %s in %s\n", free_queue_name, lib_name);
            exit(-1);
        }
    #else
        char *lib_name = "DPCTLSyclInterface.dll";
        char *malloc_name = "DPCTLmalloc_shared";
        char *free_name = "DPCTLfree_with_queue";
        char *get_queue_name = "DPCTLQueueMgr_GetCurrentQueue";
        char *free_queue_name = "DPCTLQueue_Delete";

        HMODULE sycldl = LoadLibrary(lib_name);
        assert(sycldl != NULL);
        internal_allocator = (NRT_external_malloc_func)GetProcAddress(sycldl, malloc_name);
        usmarray_allocator.malloc = save_queue_allocator;
        if (internal_allocator == NULL) {
            printf("Did not find %s in %s\n", malloc_name, lib_name);
            exit(-1);
        }

        usmarray_allocator.realloc = NULL;

        internal_free = (NRT_external_free_func)GetProcAddress(sycldl, free_name);
        usmarray_allocator.free = save_queue_deallocator;
        if (internal_free == NULL) {
            printf("Did not find %s in %s\n", free_name, lib_name);
            exit(-1);
        }

        get_queue_internal = (void *(*)(void))GetProcAddress(sycldl, get_queue_name);
        if (get_queue_internal == NULL) {
            printf("Did not find %s in %s\n", get_queue_name, lib_name);
            exit(-1);
        }
        usmarray_allocator.opaque_data = NULL;

        free_queue_internal = (void (*)(void*))GetProcAddress(sycldl, free_queue_name);
        if (free_queue_internal == NULL) {
            printf("Did not find %s in %s\n", free_queue_name, lib_name);
            exit(-1);
        }
    #endif
}

void * usmarray_get_ext_allocator(void) {
    return (void*)&usmarray_allocator;
}

static PyObject *
get_external_allocator(PyObject *self, PyObject *args) {
    return PyLong_FromVoidPtr(usmarray_get_ext_allocator());
}

static PyMethodDef ext_methods[] = {
#define declmethod_noargs(func) { #func , ( PyCFunction )func , METH_NOARGS, NULL }
    declmethod_noargs(get_external_allocator),
    {NULL},
#undef declmethod_noargs
};

static PyObject *
build_c_helpers_dict(void)
{
    PyObject *dct = PyDict_New();
    if (dct == NULL)
        goto error;

#define _declpointer(name, value) do {                 \
    PyObject *o = PyLong_FromVoidPtr(value);           \
    if (o == NULL) goto error;                         \
    if (PyDict_SetItemString(dct, name, o)) {          \
        Py_DECREF(o);                                  \
        goto error;                                    \
    }                                                  \
    Py_DECREF(o);                                      \
} while (0)

    _declpointer("usmarray_get_ext_allocator", &usmarray_get_ext_allocator);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

MOD_INIT(_dppy_rt) {
    PyObject *m;
    MOD_DEF(m, "numba_dppy._dppy_rt", "No docs", ext_methods)
    if (m == NULL)
        return MOD_ERROR_VAL;
    usmarray_memsys_init();
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
