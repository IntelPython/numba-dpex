// SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
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

#include "_dbg_printer.h"
#include "_eventstruct.h"
#include "_queuestruct.h"
#include "_usmarraystruct.h"

#include "numba/core/runtime/nrt_external.h"

// forward declarations
static struct PyUSMArrayObject *PyUSMNdArray_ARRAYOBJ(PyObject *obj);
static npy_intp product_of_shape(npy_intp *shape, npy_intp ndim);
static void *usm_device_malloc(size_t size, void *opaque_data);
static void *usm_shared_malloc(size_t size, void *opaque_data);
static void *usm_host_malloc(size_t size, void *opaque_data);
static void usm_free(void *data, void *opaque_data);
static NRT_ExternalAllocator *
NRT_ExternalAllocator_new_for_usm(DPCTLSyclQueueRef qref, size_t usm_type);
static void *DPEXRTQueue_CreateFromFilterString(const char *device);
static MemInfoDtorInfo *MemInfoDtorInfo_new(NRT_MemInfo *mi, PyObject *owner);
static NRT_MemInfo *DPEXRT_MemInfo_fill(NRT_MemInfo *mi,
                                        size_t itemsize,
                                        bool dest_is_float,
                                        bool value_is_float,
                                        int64_t value,
                                        const DPCTLSyclQueueRef qref);
static NRT_MemInfo *NRT_MemInfo_new_from_usmndarray(PyObject *ndarrobj,
                                                    void *data,
                                                    npy_intp nitems,
                                                    npy_intp itemsize,
                                                    DPCTLSyclQueueRef qref);
static NRT_MemInfo *DPEXRT_MemInfo_alloc(npy_intp size,
                                         size_t usm_type,
                                         const DPCTLSyclQueueRef qref);
static void usmndarray_meminfo_dtor(void *ptr, size_t size, void *info);
static PyObject *box_from_arystruct_parent(usmarystruct_t *arystruct,
                                           int ndim,
                                           PyArray_Descr *descr);

static int DPEXRT_sycl_usm_ndarray_from_python(PyObject *obj,
                                               usmarystruct_t *arystruct);
static PyObject *
DPEXRT_sycl_usm_ndarray_to_python_acqref(usmarystruct_t *arystruct,
                                         PyTypeObject *retty,
                                         int ndim,
                                         int writeable,
                                         PyArray_Descr *descr);
static int DPEXRT_sycl_queue_from_python(NRT_api_functions *nrt,
                                         PyObject *obj,
                                         queuestruct_t *queue_struct);
static int DPEXRT_sycl_event_from_python(NRT_api_functions *nrt,
                                         PyObject *obj,
                                         eventstruct_t *event_struct);
static PyObject *DPEXRT_sycl_queue_to_python(NRT_api_functions *nrt,
                                             queuestruct_t *queuestruct);
static PyObject *DPEXRT_sycl_event_to_python(NRT_api_functions *nrt,
                                             eventstruct_t *eventstruct);

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

/*----------------------------------------------------------------------------*/
/*--------- Functions for dpctl libsyclinterface/sycl gluing         ---------*/
/*----------------------------------------------------------------------------*/

/*!
 * @brief Creates and returns a DPCTLSyclQueueRef from a filter string.
 *
 * @param    device         A sycl::oneapi_ext::filter_string
 * @return   {DPCTLSyclQueueRef}       A DPCTLSyclQueueRef object as void*.
 */
static void *DPEXRTQueue_CreateFromFilterString(const char *device)
{
    DPCTLSyclDeviceSelectorRef dselector = NULL;
    DPCTLSyclDeviceRef dref = NULL;
    DPCTLSyclQueueRef qref = NULL;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Inside DPEXRT_get_sycl_queue %s, line %d\n", __FILE__,
        __LINE__));

    if (!(dselector = DPCTLFilterSelector_Create(device))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: Could not create a sycl::device_selector from "
            "filter string: %s at %s %d.\n",
            device, __FILE__, __LINE__));
        goto error;
    }

    if (!(dref = DPCTLDevice_CreateFromSelector(dselector)))
        goto error;

    if (!(qref = DPCTLQueue_CreateForDevice(dref, NULL, 0)))
        goto error;

    DPCTLDeviceSelector_Delete(dselector);
    DPCTLDevice_Delete(dref);

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Created sycl::queue on device %s at %s, line %d\n",
        device, __FILE__, __LINE__));

    return (void *)qref;

error:
    DPCTLDeviceSelector_Delete(dselector);
    DPCTLDevice_Delete(dref);

    return NULL;
}

static void DpexrtQueue_SubmitRange(const void *KRef,
                                    const void *QRef,
                                    void **Args,
                                    const DPCTLKernelArgType *ArgTypes,
                                    size_t NArgs,
                                    const size_t Range[3],
                                    size_t NRange,
                                    const void *DepEvents,
                                    size_t NDepEvents)
{
    DPCTLSyclEventRef eref = NULL;
    DPCTLSyclQueueRef qref = NULL;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Inside DpexrtQueue_SubmitRange %s, line %d\n", __FILE__,
        __LINE__));

    qref = (DPCTLSyclQueueRef)QRef;

    eref = DPCTLQueue_SubmitRange(
        (DPCTLSyclKernelRef)KRef, qref, Args, (DPCTLKernelArgType *)ArgTypes,
        NArgs, Range, NRange, (DPCTLSyclEventRef *)DepEvents, NDepEvents);
    DPCTLQueue_Wait(qref);
    DPCTLEvent_Wait(eref);
    DPCTLEvent_Delete(eref);

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Done with DpexrtQueue_SubmitRange %s, line %d\n",
        __FILE__, __LINE__));
}

static void DpexrtQueue_SubmitNDRange(const void *KRef,
                                      const void *QRef,
                                      void **Args,
                                      const DPCTLKernelArgType *ArgTypes,
                                      size_t NArgs,
                                      const size_t gRange[3],
                                      const size_t lRange[3],
                                      size_t Ndims,
                                      const void *DepEvents,
                                      size_t NDepEvents)
{
    DPCTLSyclEventRef eref = NULL;
    DPCTLSyclQueueRef qref = NULL;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Inside DpexrtQueue_SubmitNDRange %s, line %d\n",
        __FILE__, __LINE__));

    qref = (DPCTLSyclQueueRef)QRef;

    eref = DPCTLQueue_SubmitNDRange((DPCTLSyclKernelRef)KRef, qref, Args,
                                    (DPCTLKernelArgType *)ArgTypes, NArgs,
                                    gRange, lRange, Ndims,
                                    (DPCTLSyclEventRef *)DepEvents, NDepEvents);
    if (eref == NULL) {
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-ERROR: Kernel submission using "
                            "DpexrtQueue_SubmitNDRange failed! %s, line %d\n",
                            __FILE__, __LINE__));
    }
    else {
        DPCTLQueue_Wait(qref);
        DPCTLEvent_Wait(eref);
        DPCTLEvent_Delete(eref);
    }

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Done with DpexrtQueue_SubmitNDRange %s, line %d\n",
        __FILE__, __LINE__));
}

/*----------------------------------------------------------------------------*/
/*---------------------- Functions for NRT_MemInfo allocation ----------------*/
/*----------------------------------------------------------------------------*/

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
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-ERROR: failed to allocate memory for "
                            "NRT_ExternalAllocator at %s, line %d.\n",
                            __FILE__, __LINE__));
        goto error;
    }
    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: usm type = %d at %s, line %d.\n",
                        usm_type, __FILE__, __LINE__));

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
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-ERROR: Encountered an unknown usm "
                                     "allocation type (%d) at %s, line %d\n",
                                     usm_type, __FILE__, __LINE__));
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
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: MemInfoDtorInfo object might be corrupted. Aborting "
            "MemInfo destruction at %s, line %d\n",
            __FILE__, __LINE__));
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
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                                     "MemInfoDtorInfo object at %s, line %d\n",
                                     __FILE__, __LINE__));
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
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: Could not allocate a new NRT_MemInfo "
            "object  at %s, line %d\n",
            __FILE__, __LINE__));
        goto error;
    }

    if (!(cref = DPCTLQueue_GetContext(qref))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: Could not get the DPCTLSyclContext from "
            "the queue object at %s, line %d\n",
            __FILE__, __LINE__));
        goto error;
    }

    size_t usm_type = (size_t)DPCTLUSM_GetPointerType(data, cref);
    DPCTLContext_Delete(cref);

    // Allocate a new NRT_ExternalAllocator
    if (!(ext_alloca = NRT_ExternalAllocator_new_for_usm(qref, usm_type))) {
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                            "NRT_ExternalAllocator object  at %s, line %d\n",
                            __FILE__, __LINE__));
        goto error;
    }

    // Allocate a new MemInfoDtorInfo
    if (!(midtor_info = MemInfoDtorInfo_new(mi, ndarrobj))) {
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-ERROR: Could not allocate a new "
                                     "MemInfoDtorInfo object  at %s, line %d\n",
                                     __FILE__, __LINE__));
        goto error;
    }

    // Initialize the NRT_MemInfo object
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = usmndarray_meminfo_dtor;
    mi->dtor_info = midtor_info;
    mi->data = data;
    mi->size = nitems * itemsize;
    mi->external_allocator = ext_alloca;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: NRT_MemInfo_init mi=%p external_allocator=%p\n", mi,
        ext_alloca));

    return mi;

error:
    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-ERROR: Failed inside NRT_MemInfo_new_from_usmndarray clean up "
        "and return NULL at %s, line %d\n",
        __FILE__, __LINE__));
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
 * @param    qref         The sycl queue on which the memory was allocated. Note
 *                        that the ownership of the qref object is passed to
 *                        the NRT_MemInfo. As such, it is the caller's
 *                        responsibility to ensure the qref is nt owned by any
 *                        other object and is not deallocated. For such cases,
 *                        the caller should copy the DpctlSyclQueueRef and
 *                        pass a copy of the original qref.
 * @return   {return}     A new NRT_MemInfo object, NULL if no NRT_MemInfo
 *                        object could be created.
 */
static NRT_MemInfo *DPEXRT_MemInfo_alloc(npy_intp size,
                                         size_t usm_type,
                                         const DPCTLSyclQueueRef qref)
{
    NRT_MemInfo *mi = NULL;
    NRT_ExternalAllocator *ext_alloca = NULL;
    MemInfoDtorInfo *midtor_info = NULL;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: Inside DPEXRT_MemInfo_alloc  %s, line %d\n", __FILE__,
        __LINE__));
    // Allocate a new NRT_MemInfo object
    if (!(mi = (NRT_MemInfo *)malloc(sizeof(NRT_MemInfo)))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: Could not allocate a new NRT_MemInfo object.\n"));
        goto error;
    }

    // Allocate a new NRT_ExternalAllocator
    if (!(ext_alloca = NRT_ExternalAllocator_new_for_usm(qref, usm_type)))
        goto error;

    if (!(midtor_info = MemInfoDtorInfo_new(mi, NULL)))
        goto error;

    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = usmndarray_meminfo_dtor;
    mi->dtor_info = midtor_info;
    mi->data = ext_alloca->malloc(size, qref);

    DPEXRT_DEBUG(
        DPCTLSyclDeviceRef device_ref; device_ref = DPCTLQueue_GetDevice(qref);
        drt_debug_print(
            "DPEXRT-DEBUG: DPEXRT_MemInfo_alloc, device info in %s at %d:\n%s",
            __FILE__, __LINE__, DPCTLDeviceMgr_GetDeviceInfoStr(device_ref));
        DPCTLDevice_Delete(device_ref););

    if (mi->data == NULL)
        goto error;

    mi->size = size;
    mi->external_allocator = ext_alloca;
    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: DPEXRT_MemInfo_alloc mi=%p "
        "external_allocator=%p for usm_type=%zu on queue=%p, %s at %d\n",
        mi, ext_alloca, usm_type, DPCTLQueue_Hash(qref), __FILE__, __LINE__));

    return mi;

error:
    free(mi);
    free(ext_alloca);
    free(midtor_info);

    return NULL;
}

/**
 * @brief Interface for the core.runtime.context.DpexRTContext.meminfo_alloc.
 * This function takes an allocated memory as NRT_MemInfo and fills it with
 * the value specified by `value`.
 *
 * @param mi                An NRT_MemInfo object, should be found from memory
 *                          allocation.
 * @param itemsize          The itemsize, the size of each item in the array.
 * @param dest_is_float     True if the destination array's dtype is float.
 * @param value_is_float    True if the value to be filled is float.
 * @param value             The value to be used to fill an array.
 * @param qref              The queue on which the memory was allocated.
 * @return NRT_MemInfo*     A new NRT_MemInfo object, NULL if no NRT_MemInfo
 *                          object could be created.
 */
static NRT_MemInfo *DPEXRT_MemInfo_fill(NRT_MemInfo *mi,
                                        size_t itemsize,
                                        bool dest_is_float,
                                        bool value_is_float,
                                        int64_t value,
                                        const DPCTLSyclQueueRef qref)
{
    DPCTLSyclEventRef eref = NULL;
    size_t count = 0, size = 0, exp = 0;

    /**
     * @brief A union for bit conversion from the input int64_t value
     * to a uintX_t bit-pattern with appropriate type conversion when the
     * input value represents a float.
     */
    typedef union
    {
        float f_; /**< The float to be represented. */
        double d_;
        int8_t i8_;
        int16_t i16_;
        int32_t i32_;
        int64_t i64_;
        uint8_t ui8_;
        uint16_t ui16_;
        uint32_t ui32_; /**< The bit representation. */
        uint64_t ui64_; /**< The bit representation. */
    } bitcaster_t;

    bitcaster_t bc;
    size = mi->size;
    while (itemsize >>= 1)
        exp++;
    count = (unsigned int)(size >> exp);

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: mi->size = %u, itemsize = %u, count = %u, "
        "value = %u, Inside DPEXRT_MemInfo_fill %s, line %d\n",
        mi->size, itemsize << exp, count, value, __FILE__, __LINE__));

    if (mi->data == NULL) {
        DPEXRT_DEBUG(drt_debug_print("DPEXRT-DEBUG: mi->data is NULL, "
                                     "Inside DPEXRT_MemInfo_fill %s, line %d\n",
                                     __FILE__, __LINE__));
        goto error;
    }

    switch (exp) {
    case 3:
    {
        if (dest_is_float && value_is_float) {
            double *p = (double *)(&value);
            bc.d_ = *p;
        }
        else if (dest_is_float && !value_is_float) {
            // To stop warning: dereferencing type-punned pointer
            // will break strict-aliasing rules [-Wstrict-aliasing]
            double cd = (double)value;
            bc.d_ = *((double *)(&cd));
        }
        else if (!dest_is_float && value_is_float) {
            double *p = (double *)&value;
            bc.i64_ = (int64_t)*p;
        }
        else {
            bc.i64_ = value;
        }

        if (!(eref = DPCTLQueue_Fill64(qref, mi->data, bc.ui64_, count)))
            goto error;
        break;
    }
    case 2:
    {
        if (dest_is_float && value_is_float) {
            double *p = (double *)(&value);
            bc.f_ = (float)*p;
        }
        else if (dest_is_float && !value_is_float) {
            // To stop warning: dereferencing type-punned pointer
            // will break strict-aliasing rules [-Wstrict-aliasing]
            float cf = (float)value;
            bc.f_ = *((float *)(&cf));
        }
        else if (!dest_is_float && value_is_float) {
            double *p = (double *)&value;
            bc.i32_ = (int32_t)*p;
        }
        else {
            bc.i32_ = (int32_t)value;
        }

        if (!(eref = DPCTLQueue_Fill32(qref, mi->data, bc.ui32_, count)))
            goto error;
        break;
    }
    case 1:
    {
        if (dest_is_float)
            goto error;

        if (value_is_float) {
            double *p = (double *)&value;
            bc.i16_ = (int16_t)*p;
        }
        else {
            bc.i16_ = (int16_t)value;
        }

        if (!(eref = DPCTLQueue_Fill16(qref, mi->data, bc.ui16_, count)))
            goto error;
        break;
    }
    case 0:
    {
        if (dest_is_float)
            goto error;

        if (value_is_float) {
            double *p = (double *)&value;
            bc.i8_ = (int8_t)*p;
        }
        else {
            bc.i8_ = (int8_t)value;
        }

        if (!(eref = DPCTLQueue_Fill8(qref, mi->data, bc.ui8_, count)))
            goto error;
        break;
    }
    default:
        goto error;
    }

    DPCTLEvent_Wait(eref);
    DPCTLEvent_Delete(eref);

    return mi;

error:
    DPCTLQueue_Delete(qref);
    DPCTLEvent_Delete(eref);

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

/*----------------------------------------------------------------------------*/
/*----- Boxing and Unboxing implementations for a dpnp.ndarray PyObject ------*/
/*----------------------------------------------------------------------------*/

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
                                               usmarystruct_t *arystruct)
{
    struct PyUSMArrayObject *arrayobj = NULL;
    int i = 0, j = 0, k = 0, ndim = 0, exp = 0;
    npy_intp *shape = NULL, *strides = NULL;
    npy_intp *p = NULL, nitems;
    void *data = NULL;
    DPCTLSyclQueueRef qref = NULL;
    PyGILState_STATE gstate;
    npy_intp itemsize = 0;

    // Increment the ref count on obj to prevent CPython from garbage
    // collecting the array.
    // TODO: add extra description why do we need this
    Py_IncRef(obj);

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: In DPEXRT_sycl_usm_ndarray_from_python.\n"));

    // Check if the PyObject obj has an _array_obj attribute that is of
    // dpctl.tensor.usm_ndarray type.
    if (!(arrayobj = PyUSMNdArray_ARRAYOBJ(obj))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: PyUSMNdArray_ARRAYOBJ check failed %d\n", __FILE__,
            __LINE__));
        goto error;
    }

    if (!(ndim = UsmNDArray_GetNDim(arrayobj))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: UsmNDArray_GetNDim returned 0 at %s, line %d\n",
            __FILE__, __LINE__));
        goto error;
    }
    shape = UsmNDArray_GetShape(arrayobj);
    strides = UsmNDArray_GetStrides(arrayobj);
    data = (void *)UsmNDArray_GetData(arrayobj);
    nitems = product_of_shape(shape, ndim);
    itemsize = (npy_intp)UsmNDArray_GetElementSize(arrayobj);
    if (!(qref = UsmNDArray_GetQueueRef(arrayobj))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: UsmNDArray_GetQueueRef returned NULL at "
            "%s, line %d.\n",
            __FILE__, __LINE__));
        goto error;
    }

    if (!(arystruct->meminfo = NRT_MemInfo_new_from_usmndarray(
              obj, data, nitems, itemsize, qref)))
    {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: NRT_MemInfo_new_from_usmndarray failed "
            "at %s, line %d.\n",
            __FILE__, __LINE__));
        goto error;
    }

    arystruct->data = data;
    arystruct->sycl_queue = qref;
    arystruct->nitems = nitems;
    arystruct->itemsize = itemsize;
    arystruct->parent = obj;

    p = arystruct->shape_and_strides;

    // Calculate the exponent from the arystruct->itemsize as we know
    // itemsize is a power of two
    while (itemsize >>= 1)
        exp++;

    for (i = 0; i < ndim; ++i, ++p)
        *p = shape[i];

    // DPCTL returns a NULL pointer if the array is contiguous. dpctl stores
    // strides as number of elements and Numba stores strides as bytes, for
    // that reason we are multiplying stride by itemsize when unboxing the
    // external array.

    // FIXME: Stride computation should check order and adjust how strides are
    // calculated. Right now strides are assuming that order is C contigous.
    if (strides) {
        for (i = 0; i < ndim; ++i, ++p) {
            *p = strides[i] << exp;
        }
    }
    else {
        for (i = ndim * 2 - 1; i >= ndim; --i, ++p) {
            *p = 1;
            for (j = i, k = ndim - 1; j > ndim; --j, --k)
                *p *= shape[k];
            *p <<= exp;
        }
    }

    return 0;

error:
    // If the check failed then decrement the refcount and return an error
    // code of -1.
    // Decref the Pyobject of the array
    // ensure the GIL
    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-ERROR: Failed to unbox dpnp ndarray into a Numba "
        "arraystruct at %s, line %d\n",
        __FILE__, __LINE__));
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
static PyObject *box_from_arystruct_parent(usmarystruct_t *arystruct,
                                           int ndim,
                                           PyArray_Descr *descr)
{
    int i = 0, j = 0, k = 0, exp = 0;
    npy_intp *p = NULL;
    npy_intp *shape = NULL, *strides = NULL;
    PyObject *array = arystruct->parent;
    struct PyUSMArrayObject *arrayobj = NULL;
    npy_intp itemsize = 0;

    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: In box_from_arystruct_parent.\n"));

    if (!(arrayobj = PyUSMNdArray_ARRAYOBJ(arystruct->parent))) {
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: Arrayobj cannot be boxed from "
                            "parent as parent pointer is NULL.\n"));
        return NULL;
    }

    if ((void *)UsmNDArray_GetData(arrayobj) != arystruct->data) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-DEBUG: Arrayobj cannot be boxed "
            "from parent as data pointer in the arystruct is not the same as "
            "the data pointer in the parent object.\n"));
        return NULL;
    }

    if (UsmNDArray_GetNDim(arrayobj) != ndim)
        return NULL;

    p = arystruct->shape_and_strides;
    shape = UsmNDArray_GetShape(arrayobj);
    strides = UsmNDArray_GetStrides(arrayobj);

    // Ensure the shape of the array to be boxed matches the shape of the
    // original parent.
    for (i = 0; i < ndim; i++, p++) {
        if (shape[i] != *p)
            return NULL;
    }
    // Calculate the exponent from the arystruct->itemsize as we know
    // itemsize is a power of two
    itemsize = arystruct->itemsize;
    while (itemsize >>= 1)
        exp++;

    // Ensure the strides of the array to be boxed matches the shape of the
    // original parent. Things to note:
    //
    // 1. dpctl only stores stride information if the array has a non-unit
    // stride. If the array is unit strided then dpctl does not populate the
    // stride attribute. To verify strides, we compute the strides from the
    // shape vector.
    //
    // 2. dpctl stores strides as number of elements and Numba stores strides as
    // bytes, for that reason we are multiplying stride by itemsize when
    // unboxing the external array and dividing by itemsize when boxing the
    // array back.

    if (strides) {
        for (i = 0; i < ndim; ++i, ++p) {
            if (strides[i] << exp != *p) {
                DPEXRT_DEBUG(
                    drt_debug_print("DPEXRT-DEBUG: Arrayobj cannot be boxed "
                                    "from parent as strides in the "
                                    "arystruct are not the same as "
                                    "the strides in the parent object. "
                                    "Expected stride = %d actual stride = %d\n",
                                    strides[i] << exp, *p));
                return NULL;
            }
        }
    }
    else {
        npy_intp tmp;
        for (i = (ndim * 2) - 1; i >= ndim; --i, ++p) {
            tmp = 1;
            for (j = i, k = ndim - 1; j > ndim; --j, --k)
                tmp *= shape[k];
            tmp <<= exp;
            if (tmp != *p) {
                DPEXRT_DEBUG(
                    drt_debug_print("DPEXRT-DEBUG: Arrayobj cannot be boxed "
                                    "from parent as strides in the "
                                    "arystruct are not the same as "
                                    "the strides in the parent object. "
                                    "Expected stride = %d actual stride = %d\n",
                                    tmp, *p));
                return NULL;
            }
        }
    }

    // At the end of boxing our Meminfo destructor gets called and that will
    // decref any PyObject that was stored inside arraystruct->parent. Since,
    // we are stealing the reference and returning the original PyObject, i.e.,
    // parent, we need to increment the reference count of the parent here.
    Py_IncRef(array);

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: try_to_return_parent found a valid parent.\n"));

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
DPEXRT_sycl_usm_ndarray_to_python_acqref(usmarystruct_t *arystruct,
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
    int status = 0;
    int exp = 0;
    npy_intp itemsize = 0;

    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-DEBUG: In DPEXRT_sycl_usm_ndarray_to_python_acqref.\n"));

    if (descr == NULL) {
        PyErr_Format(
            PyExc_RuntimeError,
            "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', 'descr' is NULL");
        return MOD_ERROR_VAL;
    }

    if (!NUMBA_PyArray_DescrCheck(descr)) {
        PyErr_Format(PyExc_TypeError, "expected dtype object, got '%.200s'",
                     Py_TYPE(descr)->tp_name);
        return MOD_ERROR_VAL;
    }

    // If the arystruct has a parent attribute, try to box the parent and
    // return it.
    if (arystruct->parent) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-DEBUG: arystruct has a parent, therefore "
            "trying to box and return the parent at %s, line %d\n",
            __FILE__, __LINE__));

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
        DPEXRT_DEBUG(
            drt_debug_print("DPEXRT-DEBUG: Set the base of the boxed array "
                            "from arystruct's meminfo pointer at %s, line %d\n",
                            __FILE__, __LINE__));
        // wrap into MemInfoObject
        if (!(miobj = PyObject_New(MemInfoObject, &MemInfoType))) {
            PyErr_Format(PyExc_ValueError,
                         "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                         "failed to create a new MemInfoObject object.");
            return MOD_ERROR_VAL;
        };
        args = PyTuple_New(1);
        // PyTuple_SET_ITEM steals reference
        PyTuple_SET_ITEM(args, 0, PyLong_FromVoidPtr(arystruct->meminfo));

        //  Note: MemInfo_init() does not incref. The function steals the
        //        NRT reference, which we need to acquire.
        // Increase the refcount of the NRT_MemInfo object, i.e., mi->refct++
        NRT_MemInfo_acquire(arystruct->meminfo);
        status = MemInfo_init(miobj, args, NULL);
        if (status != 0) {
            DPEXRT_DEBUG(drt_debug_print("MemInfo_init failed at %s, line %d\n",
                                         __FILE__, __LINE__));
            Py_DECREF(args);
            PyErr_Format(PyExc_ValueError,
                         "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                         "failed to init MemInfoObject object.");
            return MOD_ERROR_VAL;
        }
        Py_DECREF(args);
    }
    else {
        PyErr_Format(PyExc_ValueError,
                     "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                     "failed to create a new MemInfoObject object since "
                     "meminfo field was null");
        return MOD_ERROR_VAL;
    }

    shape = arystruct->shape_and_strides;

    // Calculate the exponent from the arystruct->itemsize as we know
    // itemsize is a power of two
    itemsize = arystruct->itemsize;
    while (itemsize >>= 1)
        exp++;

    // Numba internally stores strides as bytes and not as elements. Divide
    // the stride by itemsize to get number of elements.
    for (size_t idx = ndim; idx < 2 * ((size_t)ndim); ++idx)
        arystruct->shape_and_strides[idx] =
            arystruct->shape_and_strides[idx] >> exp;
    strides = (shape + ndim);

    typenum = descr->type_num;
    usm_ndarr_obj = UsmNDArray_MakeFromPtr(
        ndim, shape, typenum, strides, (DPCTLSyclUSMRef)arystruct->data,
        (DPCTLSyclQueueRef)arystruct->sycl_queue, 0, (PyObject *)miobj);

    if (usm_ndarr_obj == NULL ||
        !PyObject_TypeCheck(usm_ndarr_obj, &PyUSMArrayType))
    {
        PyErr_Format(PyExc_ValueError,
                     "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                     "failed to create a new dpctl.tensor.usm_ndarray object.");
        return MOD_ERROR_VAL;
    }

    //  call new on dpnp_array
    dpnp_array_mod = PyImport_ImportModule("dpnp.dpnp_array");
    if (!dpnp_array_mod) {
        PyErr_Format(PyExc_ValueError,
                     "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                     "failed to load the dpnp.dpnp_array module.");
        return MOD_ERROR_VAL;
    }
    dpnp_array_type = PyObject_GetAttrString(dpnp_array_mod, "dpnp_array");

    if (!PyType_Check(dpnp_array_type)) {
        Py_DECREF(dpnp_array_mod);
        Py_XDECREF(dpnp_array_type);
        PyErr_Format(PyExc_ValueError,
                     "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                     "failed to crate dpnp.dpnp_array PyTypeObject.");
        return MOD_ERROR_VAL;
    }

    Py_DECREF(dpnp_array_mod);

    dpnp_array_type_obj = (PyTypeObject *)(dpnp_array_type);

    if (!(dpnp_ary = (PyObject *)dpnp_array_type_obj->tp_new(
              dpnp_array_type_obj, PyTuple_New(0), PyDict_New())))
    {
        PyErr_SetString(PyExc_ValueError,
                        "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                        "creating a dpnp.ndarray object from "
                        "a dpctl.tensor.usm_ndarray failed.");
        return MOD_ERROR_VAL;
    };

    status = PyObject_SetAttrString((PyObject *)dpnp_ary, "_array_obj",
                                    usm_ndarr_obj);
    if (status == -1) {
        Py_DECREF(dpnp_array_type_obj);
        PyErr_SetString(PyExc_TypeError,
                        "In 'DPEXRT_sycl_usm_ndarray_to_python_acqref', "
                        "could not extract '_array_obj' attribute from "
                        "dpnp.ndarray object.");
        return (PyObject *)NULL;
    }

    DPEXRT_DEBUG(drt_debug_print(
        "Returning from DPEXRT_sycl_usm_ndarray_to_python_acqref "
        "at %s, line %d\n",
        __FILE__, __LINE__));

    return (PyObject *)dpnp_ary;
}

/*----------------------------------------------------------------------------*/
/*--------------------- Box-unbox helpers for dpctl.SyclQueue       ----------*/
/*----------------------------------------------------------------------------*/

/*!
 * @brief Helper to unbox a Python dpctl.SyclQueue object to a Numba-native
 * queuestruct_t instance.
 *
 * @param    obj            A dpctl.SyclQueue Python object
 * @param    queue_struct   An instance of the struct numba-dpex uses to
 *                          represent a dpctl.SyclQueue inside Numba.
 * @return   {return}       Return code indicating success (0) or failure (-1).
 */
static int DPEXRT_sycl_queue_from_python(NRT_api_functions *nrt,
                                         PyObject *obj,
                                         queuestruct_t *queue_struct)
{
    struct PySyclQueueObject *queue_obj = NULL;
    DPCTLSyclQueueRef queue_ref = NULL;

    // We are unconditionally casting obj to a struct PySyclQueueObject*. If
    // the obj is not a struct PySyclQueueObject* then the SyclQueue_GetQueueRef
    // will error out.
    queue_obj = (struct PySyclQueueObject *)obj;

    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: In DPEXRT_sycl_queue_from_python.\n"));

    if (!(queue_ref = SyclQueue_GetQueueRef(queue_obj))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: SyclQueue_GetQueueRef returned NULL at "
            "%s, line %d.\n",
            __FILE__, __LINE__));
        goto error;
    }

    DPEXRT_DEBUG(DPCTLSyclDeviceRef device_ref;
                 device_ref = DPCTLQueue_GetDevice(queue_ref);
                 drt_debug_print("DPEXRT-DEBUG: DPEXRT_sycl_queue_from_python, "
                                 "device info in %s at %d:\n%s",
                                 __FILE__, __LINE__,
                                 DPCTLDeviceMgr_GetDeviceInfoStr(device_ref));
                 DPCTLDevice_Delete(device_ref););

    // We are doing incref here to ensure python does not release the object
    // while NRT references it. Coresponding decref is called by NRT in
    // NRT_MemInfo_pyobject_dtor once there is no reference to this object by
    // the code managed by NRT.
    Py_INCREF(queue_obj);
    queue_struct->meminfo =
        nrt->manage_memory(queue_obj, NRT_MemInfo_pyobject_dtor);
    queue_struct->queue_ref = queue_ref;

    return 0;

error:
    // If the check failed then decrement the refcount and return an error
    // code of -1.
    // Decref the Pyobject of the array
    // ensure the GIL
    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-ERROR: Failed to unbox dpctl SyclQueue into a Numba "
        "queuestruct at %s, line %d\n",
        __FILE__, __LINE__));

    return -1;
}

/*!
 * @brief A helper function that boxes a Numba-dpex queuestruct_t object into a
 * dctl.SyclQueue PyObject using the queuestruct_t's parent attribute.
 *
 * If there is no parent pointer stored in the queuestruct, then an error will
 * be raised.
 *
 * @param    queuestruct    A Numba-dpex queuestruct object.
 * @return   {return}       A PyObject created from the queuestruct->parent, if
 *                          the PyObject could not be created return NULL.
 */
static PyObject *DPEXRT_sycl_queue_to_python(NRT_api_functions *nrt,
                                             queuestruct_t *queuestruct)
{
    PyObject *queue_obj = NULL;

    queue_obj = nrt->get_data(queuestruct->meminfo);

    if (queue_obj == NULL) {
        // Make create copy of queue_ref so we don't need to manage nrt lifetime
        // from python object.
        queue_obj = SyclQueue_Make(queuestruct->queue_ref);
    }
    else {
        // Unfortunately we can not optimize (nrt->release that triggers
        // Py_DECREF() from the destructor) and Py_INCREF() because nrt may need
        // the object even if we return it to python.
        // We need to increase reference count because we are returning new
        // reference to the same queue.
        Py_INCREF(queue_obj);
    }

    // We need to release meminfo since we are taking ownership back.
    nrt->release(queuestruct->meminfo);

    return queue_obj;
}

/*----------------------------------------------------------------------------*/
/*--------------------- Box-unbox helpers for dpctl.SyclEvent       ----------*/
/*----------------------------------------------------------------------------*/

/*!
 * @brief Helper to unbox a Python dpctl.SyclEvent object to a Numba-native
 * eventstruct_t instance.
 *
 * @param    obj            A dpctl.SyclEvent Python object
 * @param    event_struct   An instance of the struct numba-dpex uses to
 *                          represent a dpctl.SyclEvent inside Numba.
 * @return   {return}       Return code indicating success (0) or failure (-1).
 */
static int DPEXRT_sycl_event_from_python(NRT_api_functions *nrt,
                                         PyObject *obj,
                                         eventstruct_t *event_struct)
{
    struct PySyclEventObject *event_obj = NULL;
    DPCTLSyclEventRef event_ref = NULL;

    // We are unconditionally casting obj to a struct PySyclEventObject*. If
    // the obj is not a struct PySyclEventObject* then the SyclEvent_GetEventRef
    // will error out.
    event_obj = (struct PySyclEventObject *)obj;

    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: In DPEXRT_sycl_event_from_python.\n"););

    if (!(event_ref = SyclEvent_GetEventRef(event_obj))) {
        DPEXRT_DEBUG(drt_debug_print(
            "DPEXRT-ERROR: SyclEvent_GetEventRef returned NULL at "
            "%s, line %d.\n",
            __FILE__, __LINE__));
        goto error;
    }

    // We are doing incref here to ensure python does not release the object
    // while NRT references it. Coresponding decref is called by NRT in
    // NRT_MemInfo_pyobject_dtor once there is no reference to this object by
    // the code managed by NRT.
    Py_INCREF(event_obj);
    event_struct->meminfo =
        nrt->manage_memory(event_obj, NRT_MemInfo_pyobject_dtor);
    event_struct->event_ref = event_ref;

    return 0;

error:
    // If the check failed then decrement the refcount and return an error
    // code of -1.
    DPEXRT_DEBUG(drt_debug_print(
        "DPEXRT-ERROR: Failed to unbox dpctl SyclEvent into a Numba "
        "eventstruct at %s, line %d\n",
        __FILE__, __LINE__));

    return -1;
}

/*!
 * @brief A helper function that boxes a Numba-dpex eventstruct_t object into a
 * dctl.SyclEvent PyObject using the eventstruct_t's parent attribute.
 *
 * If there is no parent pointer stored in the eventstruct, then an error will
 * be raised.
 *
 * @param    eventstruct    A Numba-dpex eventstruct object.
 * @return   {return}       A PyObject created from the eventstruct->parent, if
 *                          the PyObject could not be created return NULL.
 */
static PyObject *DPEXRT_sycl_event_to_python(NRT_api_functions *nrt,
                                             eventstruct_t *eventstruct)
{
    PyObject *event_obj = NULL;
    PyGILState_STATE gstate;

    event_obj = nrt->get_data(eventstruct->meminfo);

    DPEXRT_DEBUG(
        drt_debug_print("DPEXRT-DEBUG: In DPEXRT_sycl_event_to_python.\n"););

    if (event_obj == NULL) {
        // Make create copy of event_ref so we don't need to manage nrt lifetime
        // from python object.
        event_obj = SyclEvent_Make(eventstruct->event_ref);
    }
    else {
        // Unfortunately we can not optimize (nrt->release that triggers
        // Py_DECREF() from the destructor) and Py_INCREF() because nrt may need
        // the object even if we return it to python.
        // We need to increase reference count because we are returning new
        // reference to the same event.
        Py_INCREF(event_obj);
    }

    // We need to release meminfo since we are taking ownership back.
    nrt->release(eventstruct->meminfo);

    return event_obj;
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
    _declpointer("DPEXRTQueue_CreateFromFilterString",
                 &DPEXRTQueue_CreateFromFilterString);
    _declpointer("DpexrtQueue_SubmitRange", &DpexrtQueue_SubmitRange);
    _declpointer("DpexrtQueue_SubmitNDRange", &DpexrtQueue_SubmitNDRange);
    _declpointer("DPEXRT_MemInfo_alloc", &DPEXRT_MemInfo_alloc);
    _declpointer("DPEXRT_MemInfo_fill", &DPEXRT_MemInfo_fill);
    _declpointer("NRT_ExternalAllocator_new_for_usm",
                 &NRT_ExternalAllocator_new_for_usm);
    _declpointer("DPEXRT_sycl_queue_from_python",
                 &DPEXRT_sycl_queue_from_python);
    _declpointer("DPEXRT_sycl_queue_to_python", &DPEXRT_sycl_queue_to_python);
    _declpointer("DPEXRT_sycl_event_from_python",
                 &DPEXRT_sycl_event_from_python);
    _declpointer("DPEXRT_sycl_event_to_python", &DPEXRT_sycl_event_to_python);

#undef _declpointer
    return dct;
error:
    Py_XDECREF(dct);
    return NULL;
}

/*--------- Builder for the _dpexrt_python Python extension module  -- -------*/

MOD_INIT(_dpexrt_python)
{
    PyObject *m = NULL;
    PyObject *dpnp_array_type = NULL;
    PyObject *dpnp_array_mod = NULL;

    MOD_DEF(m, "_dpexrt_python", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();
    import_dpctl();

    dpnp_array_mod = PyImport_ImportModule("dpnp.dpnp_array");
    if (!dpnp_array_mod) {
        Py_DECREF(m);
        return MOD_ERROR_VAL;
    }
    dpnp_array_type = PyObject_GetAttrString(dpnp_array_mod, "dpnp_array");
    if (!PyType_Check(dpnp_array_type)) {
        Py_DECREF(m);
        Py_DECREF(dpnp_array_mod);
        Py_XDECREF(dpnp_array_type);
        return MOD_ERROR_VAL;
    }
    PyModule_AddObject(m, "dpnp_array_type", dpnp_array_type);
    Py_DECREF(dpnp_array_mod);

    PyModule_AddObject(m, "NRT_ExternalAllocator_new_for_usm",
                       PyLong_FromVoidPtr(&NRT_ExternalAllocator_new_for_usm));
    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_from_python",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_from_python));
    PyModule_AddObject(
        m, "DPEXRT_sycl_usm_ndarray_to_python_acqref",
        PyLong_FromVoidPtr(&DPEXRT_sycl_usm_ndarray_to_python_acqref));

    PyModule_AddObject(m, "DPEXRT_sycl_queue_from_python",
                       PyLong_FromVoidPtr(&DPEXRT_sycl_queue_from_python));
    PyModule_AddObject(m, "DPEXRT_sycl_queue_to_python",
                       PyLong_FromVoidPtr(&DPEXRT_sycl_queue_to_python));
    PyModule_AddObject(m, "DPEXRT_sycl_event_from_python",
                       PyLong_FromVoidPtr(&DPEXRT_sycl_event_from_python));
    PyModule_AddObject(m, "DPEXRT_sycl_event_to_python",
                       PyLong_FromVoidPtr(&DPEXRT_sycl_event_to_python));

    PyModule_AddObject(m, "DPEXRTQueue_CreateFromFilterString",
                       PyLong_FromVoidPtr(&DPEXRTQueue_CreateFromFilterString));
    PyModule_AddObject(m, "DpexrtQueue_SubmitRange",
                       PyLong_FromVoidPtr(&DpexrtQueue_SubmitRange));
    PyModule_AddObject(m, "DpexrtQueue_SubmitNDRange",
                       PyLong_FromVoidPtr(&DpexrtQueue_SubmitNDRange));
    PyModule_AddObject(m, "DPEXRT_MemInfo_alloc",
                       PyLong_FromVoidPtr(&DPEXRT_MemInfo_alloc));
    PyModule_AddObject(m, "DPEXRT_MemInfo_fill",
                       PyLong_FromVoidPtr(&DPEXRT_MemInfo_fill));
    PyModule_AddObject(m, "c_helpers", build_c_helpers_dict());
    return MOD_SUCCESS_VAL(m);
}
