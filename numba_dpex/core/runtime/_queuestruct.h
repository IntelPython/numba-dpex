#ifndef NUMBA_DPEX_QUEUESTRUCT_H_
#define NUMBA_DPEX_QUEUESTRUCT_H_
/*
 * Fill in the *queuestruct* with information from the Numpy array *obj*.
 * *queuestruct*'s layout is defined in numba.targets.arrayobj (look
 * for the ArrayTemplate class).
 */

#include <Python.h>

typedef struct
{
    PyObject *parent;
    void *queue_ref;
} queuestruct_t;

#endif /* NUMBA_DPEX_QUEUESTRUCT_H_ */
