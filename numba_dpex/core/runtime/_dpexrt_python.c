// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// A Python module that provides constructors to create a Numba MemInfo
/// PyObject using a sycl USM allocator as the external memory allocator.
/// The Module also provides the Numba box and unbox implementations for a
/// dpnp.ndarray object.
///
//===----------------------------------------------------------------------===//

#include "dpctl_capi.h"
#include "dpctl_sycl_interface.h"

#include "_meminfo_helper.h"
#include "_nrt_helper.h"
#include "_nrt_python_helper.h"

#include "numba/_arraystruct.h"

// forward declarations
static struct PyUSMArrayObject *PyUSMNdArray_ARRAYOBJ(PyObject *obj);
static PyObject *box_from_arystruct_parent(arystruct_t *arystruct,
                                           int ndim,
                                           PyArray_Descr *descr);
static PyObject *
DPEXRT_sycl_usm_ndarray_to_python_acqref(arystruct_t *arystruct,
                                         PyTypeObject *retty,
                                         int ndim,
                                         int writeable,
                                         PyArray_Descr *descr);

/*
 * Debugging printf function used internally
 */
void nrt_debug_print(char *fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/*----------------------------------------------------------------------------*/
/*--------- Helpers to get attributes out of a dpnp.ndarray PyObject ---------*/
/*----------------------------------------------------------------------------*/

/*!
 * @brief Returns the ``_array_obj`` attribute of the PyObject cast to
 * PyUSMArrayObject, if no such attribute exists returns NULL.
 *
 * @param    obj            A PyObject that will be checked for an
 *                          ``_array_obj`` attribute.
 * @return   {return}       A PyUSMArrayObject object if the input has the
 *                          ``_array_obj`` attribute, otherwise NULL.
 */
static struct PyUSMArrayObject *PyUSMNdArray_ARRAYOBJ(PyObject *obj)
{
    PyObject *arrayobj = NULL;

    arrayobj = PyObject_GetAttrString(obj, "_array_obj");

    if (!arrayobj)
        return NULL;
    if (!PyObject_TypeCheck(arrayobj, &PyUSMArrayType))
        return NULL;

    struct PyUSMArrayObject *pyusmarrayobj =
        (struct PyUSMArrayObject *)(arrayobj);

    return pyusmarrayobj;
}

/*----- Boxing and Unboxing implementations for a dpnp.ndarray PyObject ------*/

/*!
 * @brief A helper function that boxes a Numba arystruct_t object into a
 * dpnp.ndarray PyObject using the arystruct_t's parent attribute.
 *
 * @param    arystruct      A Numba arystruct_t object.
 * @param    ndim           Number of dimensions of the boxed array.
 * @param    descr          A PyArray_Desc object for the dtype of the array.
 * @return   {return}       A PyObject created from the arystruct_t->parent, if
 *                          the PyObject could not be created return NULL.
 */
static PyObject *box_from_arystruct_parent(arystruct_t *arystruct,
                                           int ndim,
                                           PyArray_Descr *descr)
{
    int i;
    npy_intp *p;
    npy_intp *shape = NULL, *strides = NULL;
    PyObject *array = arystruct->parent;
    struct PyUSMArrayObject *arrayobj = NULL;

    nrt_debug_print("DPEXRT-DEBUG: In try_to_return_parent.\n");

    if (!(arrayobj = PyUSMNdArray_ARRAYOBJ(arystruct->parent)))
        return NULL;

    if ((void *)UsmNDArray_GetData(arrayobj) != arystruct->data)
        return NULL;

    if (UsmNDArray_GetNDim(arrayobj) != ndim)
        return NULL;

    p = arystruct->shape_and_strides;
    shape = UsmNDArray_GetShape(arrayobj);
    strides = UsmNDArray_GetStrides(arrayobj);

    for (i = 0; i < ndim; i++, p++) {
        if (shape[i] != *p)
            return NULL;
    }

    if (strides) {
        if (strides[i] != *p)
            return NULL;
    }
    else {
        for (i = 1; i < ndim; ++i, ++p) {
            if (shape[i] != *p)
                return NULL;
        }
        if (*p != 1)
            return NULL;
    }

    // At the end of boxing our Meminfo destructor gets called and that will
    // decref any PyObject that was stored inside arraystruct->parent. Since,
    // we are stealing the reference and returning the original PyObject, i.e.,
    // parent, we need to increment the reference count of the parent here.
    Py_IncRef(array);

    nrt_debug_print(
        "DPEXRT-DEBUG: try_to_return_parent found a valid parent.\n");

    /* Yes, it is the same array return a new reference */
    return array;
}

/*!
 * @brief Used to implement the boxing, i.e., conversion from Numba
 * representation of a dpnp.ndarray object to a dpnp.ndarray PyObject.
 *
 * It used to steal the reference of the arystruct.
 *
 * @param arystruct The Numba internal representation of a dpnp.ndarray object.
 * @param retty Unused to be removed.
 * @param ndim is the number of dimension of the array.
 * @param writeable corresponds to the "writable" flag in the dpnp.ndarray.
 * @param descr is the data type description.
 *
 */
static PyObject *
DPEXRT_sycl_usm_ndarray_to_python_acqref(arystruct_t *arystruct,
                                         PyTypeObject *retty,
                                         int ndim,
                                         int writeable,
                                         PyArray_Descr *descr)
{
    PyObject *dpnp_ary = NULL;
    PyObject *dpnp_array_mod = NULL;
    PyObject *dpnp_array_type = NULL;
    PyObject *usm_ndarr_obj = NULL;
    PyObject *args = NULL;
    PyTypeObject *dpnp_array_type_obj = NULL;
    MemInfoObject *miobj = NULL;
    npy_intp *shape = NULL, *strides = NULL;
    int typenum = 0;

    nrt_debug_print(
        "DPEXRT-DEBUG: In DPEXRT_sycl_usm_ndarray_to_python_acqref.\n");

    if (descr == NULL) {
        PyErr_Format(
            PyExc_RuntimeError,
            "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', 'descr' is NULL");
        return NULL;
    }

    if (!NUMBA_PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_TypeError, "expected dtype object, got '%.200s'",
                     Py_TYPE(descr)->tp_name);
        return NULL;
    }

    // If the arystruct has a parent attribute, try to box the parent and
    // return it.
    if (arystruct->parent) {
        nrt_debug_print("DPEXRT-DEBUG: arystruct has a parent, therefore "
                        "trying to box and return the parent at %s, line %d\n",
                        __FILE__, __LINE__);

        PyObject *obj = box_from_arystruct_parent(arystruct, ndim, descr);
        if (obj) {
            return obj;
        }
    }

    // If the arystruct has a meminfo pointer, then use the meminfo to create
    // a MemInfoType PyTypeObject (_nrt_python_helper.h|c). The MemInfoType
    // object will then be used to create a dpctl.tensor.usm_ndarray object and
    // set as the `base` pointer of that object. The dpctl.tensor.usm_ndarray
    // object will then be used to create the final boxed dpnp.ndarray object.
    //
    // The rationale for boxing the dpnp.ndarray from the meminfo pointer is to
    // return back to Python memory that was allocated inside Numba and let
    // Python manage the lifetime of the memory.
    if (arystruct->meminfo) {
        // wrap into MemInfoObject
        miobj = PyObject_New(MemInfoObject, &MemInfoType);
        args = PyTuple_New(1);
        // PyTuple_SET_ITEM steals reference
        PyTuple_SET_ITEM(args, 0, PyLong_FromVoidPtr(arystruct->meminfo));

        NRT_Debug(nrt_debug_print(
            "NRT_adapt_ndarray_to_python arystruct->meminfo=%p\n",
            arystruct->meminfo));

        NRT_Debug(nrt_debug_print(
            "NRT_adapt_ndarray_to_python_acqref created MemInfo=%p\n", miobj));

        //  Note: MemInfo_init() does not incref. The function steals the
        //        NRT reference, which we need to acquire.
        // Increase the refcount of the NRT_MemInfo object, i.e., mi->refct++
        NRT_MemInfo_acquire(arystruct->meminfo);

        if (MemInfo_init(miobj, args, NULL)) {
            NRT_Debug(nrt_debug_print("MemInfo_init failed.\n"));
            return NULL;
        }
        Py_DECREF(args);
    }

    shape = arystruct->shape_and_strides;
    strides = shape + ndim;

    typenum = descr->type_num;
    usm_ndarr_obj = UsmNDArray_MakeFromPtr(
        ndim, shape, typenum, strides, (DPCTLSyclUSMRef)arystruct->data,
        (DPCTLSyclQueueRef)miobj->meminfo->external_allocator->opaque_data, 0,
        (PyObject *)miobj);

    if (usm_ndarr_obj == NULL ||
        !PyObject_TypeCheck(usm_ndarr_obj, &PyUSMArrayType))
    {
        return NULL;
    }

    //  call new on dpnp_array
    dpnp_array_mod = PyImport_ImportModule("dpnp.dpnp_array");
    if (!dpnp_array_mod) {
        return MOD_ERROR_VAL;
    }
    dpnp_array_type = PyObject_GetAttrString(dpnp_array_mod, "dpnp_array");

    if (!PyType_Check(dpnp_array_type)) {
        Py_DECREF(dpnp_array_mod);
        Py_XDECREF(dpnp_array_type);
        return MOD_ERROR_VAL;
    }

    Py_DECREF(dpnp_array_mod);

    dpnp_array_type_obj = (PyTypeObject *)(dpnp_array_type);

    dpnp_ary = (PyObject *)dpnp_array_type_obj->tp_new(
        dpnp_array_type_obj, PyTuple_New(0), PyDict_New());

    if (dpnp_ary == NULL) {
        nrt_debug_print("dpnp_ary==NULL \n");
    }
    else {
        nrt_debug_print("dpnp_ary=%p \n", dpnp_ary);
    }
    int status = PyObject_SetAttrString((PyObject *)dpnp_ary, "_array_obj",
                                        usm_ndarr_obj);
    nrt_debug_print("returning from status \n");
    if (status == -1) {
        nrt_debug_print("returning from status ==NULL \n");
        Py_DECREF(dpnp_array_type_obj);
        PyErr_SetString(PyExc_TypeError, "Oh no!");
        return (PyObject *)NULL;
    }
    nrt_debug_print(
        "returning from DPEXRT_sycl_usm_ndarray_to_python_acqref 1 \n");

    if (dpnp_ary == NULL) {
        nrt_debug_print(
            "returning from DPEXRT_sycl_usm_ndarray_to_python_acqref 2\n");
        return NULL;
    }

    nrt_debug_print(
        "returning from DPEXRT_sycl_usm_ndarray_to_python_acqref\n");
    return (PyObject *)dpnp_ary;
}

/*----------------------------------------------------------------------------*/
/*--------------------- The _dpexrt_python Python extension module  -- -------*/
/*----------------------------------------------------------------------------*/

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

    _declpointer("DPEXRT_sycl_usm_ndarray_to_python_acqref",
                 &DPEXRT_sycl_usm_ndarray_to_python_acqref);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

/*--------- Builder for the _dpexrt_python Python extension module  -- -------*/

MOD_INIT(_dpexrt_python)
{
    PyObject *m;
    MOD_DEF(m, "_dpexrt_python", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();
    import_dpctl();

    PyObject *dpnp_array_mod = PyImport_ImportModule("dpnp.dpnp_array");

    if (!dpnp_array_mod) {
        Py_DECREF(m);
        return MOD_ERROR_VAL;
    }
    PyObject *dpnp_array_type =
        PyObject_GetAttrString(dpnp_array_mod, "dpnp_array");
    if (!PyType_Check(dpnp_array_type)) {
        Py_DECREF(m);
        Py_DECREF(dpnp_array_mod);
        Py_XDECREF(dpnp_array_type);
        return MOD_ERROR_VAL;
    }
    PyModule_AddObject(m, "dpnp_array_type", dpnp_array_type);

    Py_DECREF(dpnp_array_mod);

    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_to_python_acqref",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_to_python_acqref));

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
