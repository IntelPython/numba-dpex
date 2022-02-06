/*
 * Helper functions for converting between native value and Python objects.
 */

#include "numba/_pymodule.h"
#include "numba/_numba_common.h"
#include "numba/core/runtime/nrt.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>

#include "_arraystruct.h"

#define SYCL_USM_ARRAY_INTERFACE "__sycl_usm_array_interface__"

typedef struct
{
    void *meminfo; /* see _rt_python.c in numba_dppy/runtime */
    PyObject *parent;
    npy_intp nitems;
    npy_intp itemsize;
    void *data;
    PyObject *syclobj; /* added for USM memory */
    npy_intp shape_and_strides[];
} sycl_usm_array_interface_t;

static int
PySyclUsmArray_Check(PyObject *obj)
{
    PyObject *dict;

    if (!PyObject_HasAttrString(obj, SYCL_USM_ARRAY_INTERFACE))
        return 0;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    if (!PyDict_Check(dict))
    {
        Py_DECREF(dict);
        return 0;
    }

    Py_DECREF(dict);
    return 1;
}

static int
size_from_shape(PyObject *shape)
{
    Py_ssize_t pos, ndim;
    PyObject *item;
    int dimsize;
    int size = 1;

    ndim = PyTuple_Size(shape);
    if (ndim < 0)
        return -1;
    if (ndim == 0)
        return 1;

    for (pos = 0; pos < ndim; ++pos)
    {
        item = PyTuple_GetItem(shape, pos);
        dimsize = PyLong_AsLong(item);

        if (dimsize <= 0)
            return -1;

        size *= dimsize;
    }

    return size;
}

static int
itemsize_from_typestr(PyObject *typestr)
{
    PyArray_Descr *descr;
    int itemsize = -1;

    if (!PyArray_DescrConverter(typestr, &descr))
    {
        return -1;
    }

    itemsize = descr->elsize;

    Py_DECREF(descr);
    return itemsize;
}

static void *
PySyclUsmArray_DATA(PyObject *obj)
{
    PyObject *dict;
    PyObject *tuple;
    PyObject *item;
    void *data = NULL;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    tuple = PyDict_GetItemString(dict, "data");

    if (PyTuple_Size(tuple) != 2)
    {
        Py_DECREF(dict);
        return NULL;
    }

    item = PyTuple_GetItem(tuple, 0);
    if (!PyLong_Check(item))
    {
        Py_DECREF(dict);
        return NULL;
    }

    data = PyLong_AsVoidPtr(item);

    Py_DECREF(dict);
    return data;
}

static PyObject *
PySyclUsmArray_SYCLOBJ(PyObject *obj)
{
    PyObject *dict;
    PyObject *syclobj;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);

    syclobj = PyDict_GetItemString(dict, "syclobj");
    Py_INCREF(syclobj);

    Py_DECREF(dict);
    return syclobj;
}

static int
PySyclUsmArray_NDIM(PyObject *obj)
{
    PyObject *dict;
    PyObject *shape;
    int ndim = 0;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    shape = PyDict_GetItemString(dict, "shape");

    ndim = PyTuple_Size(shape);

    Py_DECREF(dict);
    return ndim;
}

static int
PySyclUsmArray_DIM(PyObject *obj, int pos)
{
    PyObject *dict;
    PyObject *shape;
    PyObject *item;
    int dimsize;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    shape = PyDict_GetItemString(dict, "shape");
    item = PyTuple_GetItem(shape, pos);

    dimsize = PyLong_AsLong(item);

    Py_DECREF(dict);
    return dimsize;
}

static int
PySyclUsmArray_SIZE(PyObject *obj)
{
    PyObject *dict;
    PyObject *shape;
    int size;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    shape = PyDict_GetItemString(dict, "shape");

    size = size_from_shape(shape);

    Py_DECREF(dict);
    return size;
}

static int
PySyclUsmArray_ITEMSIZE(PyObject *obj)
{
    PyObject *dict;
    PyObject *typestr;
    int itemsize = -1;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    typestr = PyDict_GetItemString(dict, "typestr");

    itemsize = itemsize_from_typestr(typestr);

    Py_DECREF(dict);
    return itemsize;
}

static void
syclobj_dtor(void *ptr, size_t size, void *info) {
    PyGILState_STATE gstate;
    PyObject *syclobj = info;

    gstate = PyGILState_Ensure();   /* ensure the GIL */
    Py_DECREF(syclobj);            /* release the python object */
    PyGILState_Release(gstate);     /* release the GIL */
}

static void *
PySyclUsmArray_MemInfo_new(PyObject *ndary) {
    size_t dummy_size = 0;
    void *data;
    PyObject *syclobj;

    data = PySyclUsmArray_DATA(ndary);
    syclobj = PySyclUsmArray_SYCLOBJ(ndary);

    return NRT_MemInfo_new(data, dummy_size, syclobj_dtor, syclobj);
}

static int
DPPY_RT_sycl_usm_array_from_python(PyObject *obj, dp_arystruct_t *arystruct)
{
    // help: NRT_adapt_ndarray_from_python

    PyObject *ndary;
    int i, ndim;
    npy_intp *p;
    void *data;
    PyObject *syclobj;

    if (!PySyclUsmArray_Check(obj))
    {
        return -1;
    }

    ndary = obj;
    ndim = PySyclUsmArray_NDIM(ndary);
    data = PySyclUsmArray_DATA(ndary);
    syclobj = PySyclUsmArray_SYCLOBJ(ndary);

    // arystruct->meminfo = PySyclUsmArray_MemInfo_new(ndary);
    arystruct->data = data;
    arystruct->nitems = PySyclUsmArray_SIZE(ndary);
    arystruct->itemsize = PySyclUsmArray_ITEMSIZE(ndary);
    arystruct->parent = obj;
    arystruct->syclobj = syclobj;
    p = arystruct->shape_and_strides;
    for (i = 0; i < ndim; i++, p++)
    {
        *p = PySyclUsmArray_DIM(ndary, i);
    }
    for (i = 0; i < ndim; i++, p++)
    {
        // *p = PySyclUsmArray_STRIDE(ndary, i);
    }

    return 0;
}


static
PyObject* try_to_return_parent(dp_arystruct_t *arystruct, int ndim,
                               PyArray_Descr *descr)
{
    int i;
    PyObject *array = arystruct->parent;

    // if (!PyArray_Check(arystruct->parent))
    //     /* Parent is a generic buffer-providing object */
    //     goto RETURN_ARRAY_COPY;

    // if (PyArray_DATA(array) != arystruct->data)
    //     goto RETURN_ARRAY_COPY;

    // if (PyArray_NDIM(array) != ndim)
    //     goto RETURN_ARRAY_COPY;

    // if (PyObject_RichCompareBool((PyObject *) PyArray_DESCR(array),
    //                              (PyObject *) descr, Py_EQ) <= 0)
    //     goto RETURN_ARRAY_COPY;

    // for(i = 0; i < ndim; ++i) {
    //     if (PyArray_DIMS(array)[i] != arystruct->shape_and_strides[i])
    //         goto RETURN_ARRAY_COPY;
    //     if (PyArray_STRIDES(array)[i] != arystruct->shape_and_strides[ndim + i])
    //         goto RETURN_ARRAY_COPY;
    // }

    /* Yes, it is the same array
       Return new reference */
    Py_INCREF((PyObject *)array);
    return (PyObject *)array;

RETURN_ARRAY_COPY:
    return NULL;
}

/**
 * This function is used during the boxing of DPNPN ndarray type.
 * `arystruct` is a structure containing essential information from the
 *             unboxed array.
 * `retty` is the subtype of the NumPy PyArray_Type this function should return.
 *         This is related to `numba.core.types.Array.box_type`.
 * `ndim` is the number of dimension of the array.
 * `writeable` corresponds to the "writable" flag in NumPy ndarray.
 * `descr` is the NumPy data type description.
 *
 * It used to steal the reference of the arystruct.
 */
static PyObject *
DPPY_RT_sycl_usm_array_to_python_acqref(
    dp_arystruct_t *arystruct, PyTypeObject *retty,
    int ndim, int writeable, PyArray_Descr *descr)
{
    // PyArrayObject *array;
    PyObject *array;
//    MemInfoObject *miobj = NULL;
//    PyObject *args;
//     npy_intp *shape, *strides;
//     int flags = 0;

    if (descr == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "In 'DPPY_RT_sycl_usm_array_to_python_acqref', 'descr' is NULL");
        return NULL;
    }

    if (!NUMBA_PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_TypeError,
                     "expected dtype object, got '%.200s'",
                     Py_TYPE(descr)->tp_name);
        return NULL;
    }

    if (arystruct->parent) {
        PyObject *obj = try_to_return_parent(arystruct, ndim, descr);
        if (obj) {
            return obj;
        }
    }

//     if (arystruct->meminfo) {
//         /* wrap into MemInfoObject */
//         miobj = PyObject_New(MemInfoObject, &MemInfoType);
//         args = PyTuple_New(1);
//         /* SETITEM steals reference */
//         PyTuple_SET_ITEM(args, 0, PyLong_FromVoidPtr(arystruct->meminfo));
//         NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python arystruct->meminfo=%p\n", arystruct->meminfo));
//         /*  Note: MemInfo_init() does not incref.  This function steals the
//          *        NRT reference, which we need to acquire.
//          */
//         NRT_Debug(nrt_debug_print("NRT_adapt_ndarray_to_python_acqref created MemInfo=%p\n", miobj));
//         NRT_MemInfo_acquire(arystruct->meminfo);
//         if (MemInfo_init(miobj, args, NULL)) {
//             NRT_Debug(nrt_debug_print("MemInfo_init failed.\n"));
//             return NULL;
//         }
//         Py_DECREF(args);
//     }

//     shape = arystruct->shape_and_strides;
//     strides = shape + ndim;
//     Py_INCREF((PyObject *) descr);
//     array = (PyArrayObject *) PyArray_NewFromDescr(retty, descr, ndim,
//                                                    shape, strides, arystruct->data,
//                                                    flags, (PyObject *) miobj);

//     if (array == NULL)
//         return NULL;

//     /* Set writable */
// #if NPY_API_VERSION >= 0x00000007
//     if (writeable) {
//         PyArray_ENABLEFLAGS(array, NPY_ARRAY_WRITEABLE);
//     }
//     else {
//         PyArray_CLEARFLAGS(array, NPY_ARRAY_WRITEABLE);
//     }
// #else
//     if (writeable) {
//         array->flags |= NPY_WRITEABLE;
//     }
//     else {
//         array->flags &= ~NPY_WRITEABLE;
//     }
// #endif

//     if (miobj) {
//         /* Set the MemInfoObject as the base object */
// #if NPY_API_VERSION >= 0x00000007
//         if (-1 == PyArray_SetBaseObject(array,
//                                         (PyObject *) miobj))
//         {
//             Py_DECREF(array);
//             Py_DECREF(miobj);
//             return NULL;
//         }
// #else
//         PyArray_BASE(array) = (PyObject *) miobj;
// #endif

//     }
    return (PyObject *) array;
}

MOD_INIT(_rt_python)
{
    PyObject *m;
    MOD_DEF(m, "_rt_python", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    PyModule_AddObject(m, "DPPY_RT_sycl_usm_array_from_python", PyLong_FromVoidPtr(&DPPY_RT_sycl_usm_array_from_python));
    PyModule_AddObject(m, "DPPY_RT_sycl_usm_array_to_python_acqref", PyLong_FromVoidPtr(&DPPY_RT_sycl_usm_array_to_python_acqref));

    PyModule_AddObject(m, "PySyclUsmArray_Check", PyLong_FromVoidPtr(&PySyclUsmArray_Check));
    PyModule_AddObject(m, "PySyclUsmArray_NDIM", PyLong_FromVoidPtr(&PySyclUsmArray_NDIM));

    PyModule_AddObject(m, "itemsize_from_typestr", PyLong_FromVoidPtr(&itemsize_from_typestr));

    return MOD_SUCCESS_VAL(m);
}
