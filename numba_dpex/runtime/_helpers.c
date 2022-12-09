static int size_from_shape(PyObject *shape)
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

    for (pos = 0; pos < ndim; ++pos) {
        item = PyTuple_GetItem(shape, pos);
        dimsize = PyLong_AsLong(item);

        if (dimsize <= 0)
            return -1;

        size *= dimsize;
    }

    return size;
}

static int itemsize_from_typestr(PyObject *typestr)
{
    PyArray_Descr *descr;
    int itemsize = -1;

    if (!PyArray_DescrConverter(typestr, &descr)) {
        return -1;
    }

    itemsize = descr->elsize;

    Py_DECREF(descr);
    return itemsize;
}
