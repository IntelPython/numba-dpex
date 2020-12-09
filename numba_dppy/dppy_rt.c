#include "_pymodule.h"
#include "core/runtime/nrt_external.h"
#include "assert.h"
#include <dlfcn.h>
#include <stdio.h>

NRT_ExternalAllocator usmarray_allocator;

void usmarray_memsys_init(void) {
    void *(*get_queue)(void);
    char *lib_name = "libDPCTLSyclInterface.so";
    char *malloc_name = "DPCTLmalloc_shared";
    char *free_name = "DPCTLfree_with_queue";
    char *get_queue_name = "DPCTLQueueMgr_GetCurrentQueue";

    void *sycldl = dlopen(lib_name, RTLD_NOW);
    assert(sycldl != NULL);
    usmarray_allocator.malloc = (NRT_external_malloc_func)dlsym(sycldl, malloc_name);
    if (usmarray_allocator.malloc == NULL) {
        printf("Did not find %s in %s\n", malloc_name, lib_name);
        exit(-1);
    }
    usmarray_allocator.realloc = NULL;
    usmarray_allocator.free = (NRT_external_free_func)dlsym(sycldl, free_name);
    if (usmarray_allocator.free == NULL) {
        printf("Did not find %s in %s\n", free_name, lib_name);
        exit(-1);
    }
    get_queue = (void *(*))dlsym(sycldl, get_queue_name);
    if (get_queue == NULL) {
        printf("Did not find %s in %s\n", get_queue_name, lib_name);
        exit(-1);
    }
    usmarray_allocator.opaque_data = get_queue();
}

void * usmarray_get_ext_allocator(void) {
    printf("usmarray_get_ext_allocator %p\n", &usmarray_allocator);
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
