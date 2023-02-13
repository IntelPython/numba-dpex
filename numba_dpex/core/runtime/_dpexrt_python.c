// SPDX-FileCopyrightText: 2020 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
///
/// \file
/// A Python module that pprovides constructors to create a Numba MemInfo
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
static npy_intp product_of_shape(npy_intp *shape, npy_intp ndim);
static void *usm_device_malloc(size_t size, void *opaque_data);
static void *usm_shared_malloc(size_t size, void *opaque_data);
static void *usm_host_malloc(size_t size, void *opaque_data);
static void usm_free(void *data, void *opaque_data);
static NRT_ExternalAllocator *
NRT_ExternalAllocator_new_for_usm(DPCTLSyclQueueRef qref, size_t usm_type);
static MemInfoDtorInfo *MemInfoDtorInfo_new(NRT_MemInfo *mi, PyObject *owner);
static NRT_MemInfo *NRT_MemInfo_new_from_usmndarray(PyObject *ndarrobj,
                                                    void *data,
                                                    npy_intp nitems,
                                                    npy_intp itemsize,
                                                    DPCTLSyclQueueRef qref);
static void usmndarray_meminfo_dtor(void *ptr, size_t size, void *info);
static PyObject *box_from_arystruct_parent(arystruct_t *arystruct,
                                           int ndim,
                                           PyArray_Descr *descr);

static int DPEXRT_sycl_usm_ndarray_from_python(PyObject *obj,
                                               arystruct_t *arystruct);
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

/** An NRT_external_malloc_func implementation using DPCTLmalloc_device.
 *
 */
static void *usm_device_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_device(size, qref);
}

/** An NRT_external_malloc_func implementation using DPCTLmalloc_shared.
 *
 */
static void *usm_shared_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_shared(size, qref);
}

/** An NRT_external_malloc_func implementation using DPCTLmalloc_host.
 *
 */
static void *usm_host_malloc(size_t size, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;

    qref = (DPCTLSyclQueueRef)opaque_data;
    return DPCTLmalloc_host(size, qref);
}

/** An NRT_external_free_func implementation based on DPCTLfree_with_queue
 *
 */
static void usm_free(void *data, void *opaque_data)
{
    DPCTLSyclQueueRef qref = NULL;
    qref = (DPCTLSyclQueueRef)opaque_data;

    DPCTLfree_with_queue(data, qref);
}

/*!
 * @brief Creates a new NRT_ExternalAllocator object tied to a SYCL USM
 *        allocator.
 *
 * @param    qref           A DPCTLSyclQueueRef opaque pointer for a sycl queue.
 * @param    usm_type       Indicates the type of usm allocator to use.
 *                          - 1: device
 *                          - 2: shared
 *                          - 3: host
 *                          The values are as defined in the DPCTLSyclUSMType
 *                          enum in dpctl's libsyclinterface library.
 * @return   {return}       A new NRT_ExternalAllocator object or NULL if
 *                          object creation failed.
 */
static NRT_ExternalAllocator *
NRT_ExternalAllocator_new_for_usm(DPCTLSyclQueueRef qref, size_t usm_type)
{

    NRT_ExternalAllocator *allocator = NULL;

    allocator = (NRT_ExternalAllocator *)malloc(sizeof(NRT_ExternalAllocator));
    if (allocator == NULL) {
        nrt_debug_print("DPEXRT-ERROR: failed to allocate memory for "
                        "NRT_ExternalAllocator at %s, line %d.\n",
                        __FILE__, __LINE__);
        goto error;
    }
    nrt_debug_print("DPEXRT-DEBUG: usm type = %d at %s, line %d.\n", usm_type,
                    __FILE__, __LINE__);

    switch (usm_type) {
    case 1:
        allocator->malloc = usm_device_malloc;
        break;
    case 2:
        allocator->malloc = usm_shared_malloc;
        break;
    case 3:
        allocator->malloc = usm_host_malloc;
        break;
    default:
        nrt_debug_print("DPEXRT-ERROR: Encountered an unknown usm "
                        "allocation type (%d) at %s, line %d\n",
                        usm_type, __FILE__, __LINE__);
        goto error;
    }

    allocator->realloc = NULL;
    allocator->free = usm_free;
    allocator->opaque_data = (void *)qref;

    return allocator;

error:
    free(allocator);
    return NULL;
}

/*!
 * @brief  Destructor function for a MemInfo object allocated inside DPEXRT. The
 * destructor is called by Numba using the NRT_MemInfo_release function.
 *
 * The destructor does the following clean up:
 *     - Frees the data associated with the MemInfo object if there was no
 *       parent PyObject that owns the data.
 *     - Frees the DpctlSyclQueueRef pointer stored in the opaque data of the
 *       MemInfo's external_allocator member.
 *     - Frees the external_allocator object associated with the MemInfo object.
 *     - If there was a PyObject associated with the MemInfo, then
 *       the reference count on that object.
 *     - Frees the MemInfoDtorInfo wrapper object that was stored as the
 *       dtor_info member of the MemInfo.
 *
 * @param    ptr            *Unused*, the argument is required to match the
 *                          type of the NRT_dtor_function pointer type.
 * @param    size           *Unused*, the argument is required to match the
 *                          type of the NRT_dtor_function pointer type.
 * @param    info           A MemInfoDtorInfo object that stores a reference to
 *                          the parent meminfo and any original PyObject from
 *                          which the meminfo was created.
 */
static void usmndarray_meminfo_dtor(void *ptr, size_t size, void *info)
{
    MemInfoDtorInfo *mi_dtor_info = NULL;

    // Sanity-check to make sure the mi_dtor_info is an actual pointer.
    if (!(mi_dtor_info = (MemInfoDtorInfo *)info)) {
        nrt_debug_print(
            "DPEXRT-ERROR: MemInfoDtorInfo object might be corrupted. Aborting "
            "MemInfo destruction at %s, line %d\n",
            __FILE__, __LINE__);
        return;
    }

    // If there is no owner PyObject, free the data by calling the
    // external_allocator->free
    if (!(mi_dtor_info->owner))
        mi_dtor_info->mi->external_allocator->free(
            mi_dtor_info->mi->data,
            mi_dtor_info->mi->external_allocator->opaque_data);

    // free the DpctlSyclQueueRef object stored inside the external_allocator
    DPCTLQueue_Delete(
        (DPCTLSyclQueueRef)mi_dtor_info->mi->external_allocator->opaque_data);

    // free the external_allocator object
    free(mi_dtor_info->mi->external_allocator);

    // Set the pointer to NULL to prevent NRT_dealloc trying to use it free
    // the meminfo object
    mi_dtor_info->mi->external_allocator = NULL;

    if (mi_dtor_info->owner) {
        // Decref the Pyobject from which the MemInfo was created
        PyGILState_STATE gstate;
        PyObject *ownerobj = mi_dtor_info->owner;
        // ensure the GIL
        gstate = PyGILState_Ensure();
        // decref the python object
        Py_DECREF(ownerobj);
        // release the GIL
        PyGILState_Release(gstate);
    }

    // Free the MemInfoDtorInfo object
    free(mi_dtor_info);
}

/*!
 * @brief  Allocates and returns a new MemInfoDtorInfo object.
 *
 * @param    mi             The parent NRT_MemInfo object for which the
 *                          dtor_info attribute is being created.
 * @param    owner          A PyObject from which the NRT_MemInfo object was
 *                          created, maybe NULL if no such object exists.
 * @return   {return}       A new MemInfoDtorInfo object.
 */
static MemInfoDtorInfo *MemInfoDtorInfo_new(NRT_MemInfo *mi, PyObject *owner)
{
    MemInfoDtorInfo *mi_dtor_info = NULL;

    if (!(mi_dtor_info = (MemInfoDtorInfo *)malloc(sizeof(MemInfoDtorInfo)))) {
        nrt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                        "MemInfoDtorInfo object at %s, line %d\n",
                        __FILE__, __LINE__);
        return NULL;
    }
    mi_dtor_info->mi = mi;
    mi_dtor_info->owner = owner;

    return mi_dtor_info;
}

/*!
 * @brief Creates a NRT_MemInfo object for a dpnp.ndarray
 *
 * @param    ndarrobj       An dpnp.ndarray PyObject
 * @param    data           The data pointer of the dpnp.ndarray
 * @param    nitems         The number of elements in the dpnp.ndarray.
 * @param    itemsize       The size of each element of the dpnp.ndarray.
 * @param    qref           A SYCL queue pointer wrapper on which the memory
 *                          of the dpnp.ndarray was allocated.
 * @return   {return}       A new NRT_MemInfo object
 */
static NRT_MemInfo *NRT_MemInfo_new_from_usmndarray(PyObject *ndarrobj,
                                                    void *data,
                                                    npy_intp nitems,
                                                    npy_intp itemsize,
                                                    DPCTLSyclQueueRef qref)
{
    NRT_MemInfo *mi = NULL;
    NRT_ExternalAllocator *ext_alloca = NULL;
    MemInfoDtorInfo *midtor_info = NULL;
    DPCTLSyclContextRef cref = NULL;

    // Allocate a new NRT_MemInfo object
    if (!(mi = (NRT_MemInfo *)malloc(sizeof(NRT_MemInfo)))) {
        nrt_debug_print("DPEXRT-ERROR: Could not allocate a new NRT_MemInfo "
                        "object  at %s, line %d\n",
                        __FILE__, __LINE__);
        goto error;
    }

    if (!(cref = DPCTLQueue_GetContext(qref))) {
        nrt_debug_print("DPEXRT-ERROR: Could not get the DPCTLSyclContext from "
                        "the queue object at %s, line %d\n",
                        __FILE__, __LINE__);
        goto error;
    }

    size_t usm_type = (size_t)DPCTLUSM_GetPointerType(data, cref);
    DPCTLContext_Delete(cref);

    // Allocate a new NRT_ExternalAllocator
    if (!(ext_alloca = NRT_ExternalAllocator_new_for_usm(qref, usm_type))) {
        nrt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                        "NRT_ExternalAllocator object  at %s, line %d\n",
                        __FILE__, __LINE__);
        goto error;
    }

    // Allocate a new MemInfoDtorInfo
    if (!(midtor_info = MemInfoDtorInfo_new(mi, ndarrobj))) {
        nrt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                        "MemInfoDtorInfo object  at %s, line %d\n",
                        __FILE__, __LINE__);
        goto error;
    }

    // Initialize the NRT_MemInfo object
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = usmndarray_meminfo_dtor;
    mi->dtor_info = midtor_info;
    mi->data = data;
    mi->size = nitems * itemsize;
    mi->external_allocator = ext_alloca;

    nrt_debug_print(
        "DPEXRT-DEBUG: NRT_MemInfo_init mi=%p external_allocator=%p\n", mi,
        ext_alloca);

    return mi;

error:
    nrt_debug_print(
        "DPEXRT-ERROR: Failed inside NRT_MemInfo_new_from_usmndarray clean up "
        "and return NULL at %s, line %d\n",
        __FILE__, __LINE__);
    free(mi);
    free(ext_alloca);
    return NULL;
}

/*!
 * @brief Creates a NRT_MemInfo object whose data is allocated using a USM
 * allocator.
 *
 * @param    size         The size of memory (data) owned by the NRT_MemInfo
 *                        object.
 * @param    usm_type     The usm type of the memory.
 * @param    device       The device on which the memory was allocated.
 * @return   {return}     A new NRT_MemInfo object, NULL if no NRT_MemInfo
 *                        object could be created.
 */
static NRT_MemInfo *
DPEXRT_MemInfo_alloc(npy_intp size, size_t usm_type, const char *device)
{
    NRT_MemInfo *mi = NULL;
    NRT_ExternalAllocator *ext_alloca = NULL;
    MemInfoDtorInfo *midtor_info = NULL;
    DPCTLSyclDeviceSelectorRef dselector = NULL;
    DPCTLSyclDeviceRef dref = NULL;
    DPCTLSyclQueueRef qref = NULL;

    nrt_debug_print("DPEXRT-DEBUG: Inside DPEXRT_MemInfo_alloc  %s, line %d\n",
                    __FILE__, __LINE__);
    // Allocate a new NRT_MemInfo object
    if (!(mi = (NRT_MemInfo *)malloc(sizeof(NRT_MemInfo)))) {
        nrt_debug_print(
            "DPEXRT-ERROR: Could not allocate a new NRT_MemInfo object.\n");
        goto error;
    }

    if (!(dselector = DPCTLFilterSelector_Create(device))) {
        nrt_debug_print(
            "DPEXRT-ERROR: Could not create a sycl::device_selector from "
            "filter string: %s at %s %d.\n",
            device, __FILE__, __LINE__);
        goto error;
    }

    if (!(dref = DPCTLDevice_CreateFromSelector(dselector)))
        goto error;

    if (!(qref = DPCTLQueue_CreateForDevice(dref, NULL, 0)))
        goto error;

    DPCTLDeviceSelector_Delete(dselector);
    DPCTLDevice_Delete(dref);

    // Allocate a new NRT_ExternalAllocator
    if (!(ext_alloca = NRT_ExternalAllocator_new_for_usm(qref, usm_type)))
        goto error;

    if (!(midtor_info = MemInfoDtorInfo_new(mi, NULL)))
        goto error;

    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = usmndarray_meminfo_dtor;
    mi->dtor_info = midtor_info;
    mi->data = ext_alloca->malloc(size, qref);

    if (mi->data == NULL)
        goto error;

    mi->size = size;
    mi->external_allocator = ext_alloca;
    nrt_debug_print(
        "DPEXRT-DEBUG: DPEXRT_MemInfo_alloc mi=%p "
        "external_allocator=%p for usm_type %zu on device %s, %s at %d\n",
        mi, ext_alloca, usm_type, device, __FILE__, __LINE__);

    return mi;

error:
    free(mi);
    free(ext_alloca);
    free(midtor_info);
    DPCTLDeviceSelector_Delete(dselector);
    DPCTLDevice_Delete(dref);

    return NULL;
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

/*!
 * @brief Returns the product of the elements in an array of a given
 * length.
 *
 * @param    shape          An array of integers
 * @param    ndim           The length of the ``shape`` array.
 * @return   {return}       The product of the elements in the ``shape`` array.
 */
static npy_intp product_of_shape(npy_intp *shape, npy_intp ndim)
{
    npy_intp nelems = 1;

    for (int i = 0; i < ndim; ++i)
        nelems *= shape[i];

    return nelems;
}

/*----- Boxing and Unboxing implementations for a dpnp.ndarray PyObject ------*/

/*!
 * @brief Unboxes a PyObject that may represent a dpnp.ndarray into a Numba
 * native represetation.
 *
 * @param    obj            A Python object that may be a dpnp.ndarray
 * @param    arystruct      Numba's internal native represnetation for a given
 *                          instance of a dpnp.ndarray
 * @return   {return}       Error code representing success (0) or failure (-1).
 */
static int DPEXRT_sycl_usm_ndarray_from_python(PyObject *obj,
                                               arystruct_t *arystruct)
{
    struct PyUSMArrayObject *arrayobj = NULL;
    int i, ndim;
    npy_intp *shape = NULL, *strides = NULL;
    npy_intp *p = NULL, nitems, itemsize;
    void *data = NULL;
    DPCTLSyclQueueRef qref = NULL;
    PyGILState_STATE gstate;

    // Increment the ref count on obj to prevent CPython from garbage
    // collecting the array.
    Py_IncRef(obj);

    nrt_debug_print("DPEXRT-DEBUG: In DPEXRT_sycl_usm_ndarray_from_python.\n");

    // Check if the PyObject obj has an _array_obj attribute that is of
    // dpctl.tensor.usm_ndarray type.
    if (!(arrayobj = PyUSMNdArray_ARRAYOBJ(obj))) {
        nrt_debug_print("DPEXRT-ERROR: PyUSMNdArray_ARRAYOBJ check failed %d\n",
                        __FILE__, __LINE__);
        goto error;
    }

    if (!(ndim = UsmNDArray_GetNDim(arrayobj))) {
        nrt_debug_print(
            "DPEXRT-ERROR: UsmNDArray_GetNDim returned 0 at %s, line %d\n",
            __FILE__, __LINE__);
        goto error;
    }
    shape = UsmNDArray_GetShape(arrayobj);
    strides = UsmNDArray_GetStrides(arrayobj);
    data = (void *)UsmNDArray_GetData(arrayobj);
    nitems = product_of_shape(shape, ndim);
    itemsize = (npy_intp)UsmNDArray_GetElementSize(arrayobj);
    if (!(qref = UsmNDArray_GetQueueRef(arrayobj))) {
        nrt_debug_print("DPEXRT-ERROR: UsmNDArray_GetQueueRef returned NULL at "
                        "%s, line %d.\n",
                        __FILE__, __LINE__);
        goto error;
    }
    else {
        nrt_debug_print("qref addr : %p\n", qref);
    }

    if (!(arystruct->meminfo = NRT_MemInfo_new_from_usmndarray(
              obj, data, nitems, itemsize, qref)))
    {
        nrt_debug_print("DPEXRT-ERROR: NRT_MemInfo_new_from_usmndarray failed "
                        "at %s, line %d.\n",
                        __FILE__, __LINE__);
        goto error;
    }

    arystruct->data = data;
    arystruct->nitems = nitems;
    arystruct->itemsize = itemsize;
    arystruct->parent = obj;

    p = arystruct->shape_and_strides;

    for (i = 0; i < ndim; ++i, ++p)
        *p = shape[i];

    // DPCTL returns a NULL pointer if the array is contiguous
    // FIXME: Stride computation should check order and adjust how strides are
    // calculated. Right now strides are assuming that order is C contigous.
    if (strides) {
        for (i = 0; i < ndim; ++i, ++p) {
            *p = strides[i];
        }
    }
    else {
        for (i = 1; i < ndim; ++i, ++p) {
            *p = shape[i];
        }
        *p = 1;
    }

    // --- DEBUG
    nrt_debug_print("DPEXRT-DEBUG: Assigned shape_and_strides at %s, line %d\n",
                    __FILE__, __LINE__);
    p = arystruct->shape_and_strides;
    for (i = 0; i < ndim * 2; ++i, ++p) {
        nrt_debug_print("DPEXRT-DEBUG: arraystruct->p[%d] = %d, ", i, *p);
    }
    nrt_debug_print("\n");
    // -- DEBUG

    return 0;

error:
    // If the check failed then decrement the refcount and return an error
    // code of -1.
    // Decref the Pyobject of the array
    // ensure the GIL
    nrt_debug_print("DPEXRT-ERROR: Failed to unbox dpnp ndarray into a Numba "
                    "arraystruct at %s, line %d\n",
                    __FILE__, __LINE__);
    gstate = PyGILState_Ensure();
    // decref the python object
    Py_DECREF(obj);
    // release the GIL
    PyGILState_Release(gstate);

    return -1;
}

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

    _declpointer("DPEXRT_sycl_usm_ndarray_from_python",
                 &DPEXRT_sycl_usm_ndarray_from_python);
    _declpointer("DPEXRT_sycl_usm_ndarray_to_python_acqref",
                 &DPEXRT_sycl_usm_ndarray_to_python_acqref);
    _declpointer("DPEXRT_MemInfo_alloc", &DPEXRT_MemInfo_alloc);
    _declpointer("NRT_ExternalAllocator_new_for_usm",
                 &NRT_ExternalAllocator_new_for_usm);

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
    static PyTypeObject *dpnp_array_type_obj;
    dpnp_array_type_obj = (PyTypeObject *)(dpnp_array_type);

    PyModule_AddObject(m, "NRT_ExternalAllocator_new_for_usm",
                       PyLong_FromVoidPtr(&NRT_ExternalAllocator_new_for_usm));
    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_from_python",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_from_python));
    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_to_python_acqref",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_to_python_acqref));
    PyModule_AddObject(m, "DPEXRT_MemInfo_alloc",
                       PyLong_FromVoidPtr(&DPEXRT_MemInfo_alloc));

    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
