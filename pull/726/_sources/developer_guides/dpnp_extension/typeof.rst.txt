Infer Numba type
````````````````

The first step is to infer, at call-time, a Numba type for each of the
function’s concrete arguments.
See `Type resolution <https://numba.readthedocs.io/en/stable/developer/dispatching.html#type-resolution>`_.

Numba need to understand type of input objects.
See `@typeof_impl.register <https://numba.readthedocs.io/en/stable/extending/low-level.html#typeof_impl.register>`_.

It inspect the object and, based on its Python type, query various properties to
infer the appropriate Numba type.

Simple Python types usually infers to simple Numba types.
Complex Python types (i.e. tuples or arrays) can infer to multiple Numba types.

Infer Numba type for DPNP array (``@typeof_impl.register``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  # numba_dppy/types/dpnp_typeof.py
  @typeof_impl.register(dpnp.ndarray)
  def typeof_dpnp_ndarray(val, c):
    # query properties from val
    ...
    return dpnp_ndarray_Type(...)
