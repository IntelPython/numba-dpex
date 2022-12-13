Boxing and unboxing
```````````````````

See documentation for low-level API `Boxing and unboxing https://numba.readthedocs.io/en/stable/extending/low-level.html#boxing-and-unboxing`_.
See example `Boxing and unboxing <https://numba.readthedocs.io/en/stable/extending/interval-example.html#boxing-and-unboxing>`_.

Unbox is the other name for “convert a Python object to a native value” (it fits
the idea of a Python object as a sophisticated box containing a simple native
value).

Boxing and unboxing for DPNP array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unboxing for ``dpnp_ndarray_Type`` fallbacks to unboxing for ``types.Array``.

.. code-block:: python

  # numba_dpex/types/dpnp_boxing.py
  @unbox(dpnp_ndarray_Type)
  def unbox_array(typ, obj, c):
    # using c.pyapi for generating Python API calls
    # for extracting properties from obj value
    # using c.builder for generating code
    # for conversion and creating native value
    ...
    # 1. make LLVM type from typ and generate code for creating native structure
    nativearycls = c.context.make_array(typ)
    nativeary = nativearycls(c.context, c.builder)

    # 2. get LLVM variable
    aryptr = nativeary._getpointer()

    # 3. generating code for checking conversion
    failed = c.builder.or_(...)

    # 4. return native value and error
    return NativeValue(c.builder.load(aryptr), is_error=failed)

.. code-block:: python

  # numba/core/cgutils.py
  class _StructProxy(object):
    ...
  class ValueStructProxy(_StructProxy):
    ...

  # numba/np/arrayobj.py
  def make_array(array_type):
    ...
    # class ValueStructProxy_dpnp.ndarray(ValueStructProxy):
    #   _fe_type = ...
    #   ...
    base = cgutils.create_struct_proxy(real_array_type)
    ...
    class ArrayStruct(ValueStructProxy_dpnp.ndarray):
      ...
      def _make_refs():

      def shape():
      ...

Unboxing generates call to ``NRT_adapt_ndarray_from_python``.
It checks that Python object is a NumPy array.

.. code-block:: c
  // numba/core/runtime/_nrt_python.c
  NRT_adapt_ndarray_from_python(...) {
    ...
    if (!PyArray_Check(obj)) {
        return -1;
    }
    ...
  }

.. code-block:: python
  # numba/core/pythonapi.py
  def nrt_adapt_ndarray_from_python(self, ary, ptr):
    ...
    fn = self._get_function(fnty, name="NRT_adapt_ndarray_from_python")
    ...
    return self.builder.call(fn, (ary, ptr))

  # numba_dpex/types/dpnp_boxing.py
  @unbox(dpnp_ndarray_Type)
  def unbox_array(typ, obj, c):
    ...
    if c.context.enable_nrt:
      errcode = c.pyapi.nrt_adapt_ndarray_from_python(obj, ptr)
    ...

It calls `PyArray_Check <https://numpy.org/doc/stable/reference/c-api/array.html?highlight=pyarray_check#c.PyArray_Check>`_.
Object should be a subclass of NumPy array.
dpnp.ndarray is not a subclass of NumPy array.
We need another NRT function.
dpctl ndarray support was assuming it is a subclass of NumPy array.

.. also::
  See class ``ArrayCompatible`` in :file:`numba/core/types/abstract.py`.
  It is a Numba type for objects with __array__ function.

``make_array()`` receives real array type from ``types.Array`` subclass.
.. code-block:: python
  # numba/np/arrayobj.py
  def make_array(array_type):
    real_array_type = array_type.as_array
    ...

``ArrayStruct._make_refs()`` generates call for ``__array__``.
.. code-block:: python
  # numba/np/arrayobj.py
  class ArrayStruct(base):
    def _make_refs(self, ref):
      ...
      try:
          array_impl = self._context.get_function('__array__', sig)
      except NotImplementedError:
          return super(ArrayStruct, self)._make_refs(ref)
      ...
      ref = array_impl(self._builder, (outer_ref,))

.. question::
  ``dpnp.ndarray`` will suppirt ``__array__``? How it will work? Copy data to
  host?

Pseudo code for unboxing:
.. code-block:: python
  def unboxing(array):
    real_array = array.__array__()
    PyArray_Check(real_array)
    native_value = adapt_ndarray_from_python(real_array)
    return native_value
