Define data model
`````````````````

See exemple `Defining the data model for native intervals <https://numba.readthedocs.io/en/stable/extending/interval-example.html#defining-the-data-model-for-native-intervals>`_.

Each type has to define a tailored native representation, also called a data
model.

Define data model for DPNP array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse Array model.
Additional information like adress space is in Numba type.

.. todo::
  Where is queue and device?

.. code-block:: python

  # numba_dppy/types/dpnp_models.py
  from numba.core.datamodel.models import ArrayModel
  from numba.extending import register_model

  register_model(dpnp_ndarray_Type)(ArrayModel)

  # numba/core/datamodel/models.py
  @register_default(types.Array)
  @register_default(types.Buffer)
  @register_default(types.ByteArray)
  @register_default(types.Bytes)
  @register_default(types.MemoryView)
  @register_default(types.PyArray)
  class ArrayModel(StructModel):
    ...
    def __init__(self, ...):
      ...
      members = [
        ('meminfo', types.MemInfoPointer(fe_type.dtype)),
        ('parent', types.pyobject),
        ('nitems', types.intp),
        ('itemsize', types.intp),
        ('data', types.CPointer(fe_type.dtype)),
        ('shape', types.UniTuple(types.intp, ndim)),
        ('strides', types.UniTuple(types.intp, ndim)),
      ]
      ...

  class StructModel(CompositeModel):
    ...

  class CompositeModel(DataModel):
    ...

  class DataModel(object):
    ...
