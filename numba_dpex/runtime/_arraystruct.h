#ifndef NUMBA_DP_ARYSTRUCT_H_
#define NUMBA_DP_ARYSTRUCT_H_
/*
 * Fill in the *arystruct* with information from the DPNP or USM array *obj*.
 * *dp_arystruct*'s layout is defined in numba_dpex.core.types.dpnp_models (look
 * for the dpnp_ndarray_Model class).
 */

typedef struct
{
    void *meminfo; /* see _rt_python.c in numba_dpex/runtime */
    PyObject *parent;
    npy_intp nitems;
    npy_intp itemsize;
    void *data;
    PyObject *syclobj;
    npy_intp shape_and_strides[];
} dp_arystruct_t;

#endif /* NUMBA_DP_ARYSTRUCT_H_ */
