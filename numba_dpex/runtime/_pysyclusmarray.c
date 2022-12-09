
static int PySyclUsmArray_Check(PyObject *obj)
{
    PyObject *dict;

    if (!PyObject_HasAttrString(obj, SYCL_USM_ARRAY_INTERFACE))
        return 0;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    if (!PyDict_Check(dict)) {
        Py_DECREF(dict);
        return 0;
    }

    Py_DECREF(dict);
    return 1;
}

static void *PySyclUsmArray_DATA(PyObject *obj)
{
    PyObject *dict;
    PyObject *tuple;
    PyObject *item;
    void *data = NULL;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);
    tuple = PyDict_GetItemString(dict, "data");

    if (PyTuple_Size(tuple) != 2) {
        Py_DECREF(dict);
        return NULL;
    }

    item = PyTuple_GetItem(tuple, 0);
    if (!PyLong_Check(item)) {
        Py_DECREF(dict);
        return NULL;
    }

    data = PyLong_AsVoidPtr(item);

    Py_DECREF(dict);
    return data;
}

static PyObject *PySyclUsmArray_SYCLOBJ(PyObject *obj)
{
    PyObject *dict;
    PyObject *syclobj;

    dict = PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE);

    syclobj = PyDict_GetItemString(dict, "syclobj");
    Py_INCREF(syclobj);

    Py_DECREF(dict);
    return syclobj;
}

static int PySyclUsmArray_NDIM(PyObject *obj)
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

static int PySyclUsmArray_DIM(PyObject *obj, int pos)
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

static int PySyclUsmArray_SIZE(PyObject *obj)
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

static int PySyclUsmArray_ITEMSIZE(PyObject *obj)
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
