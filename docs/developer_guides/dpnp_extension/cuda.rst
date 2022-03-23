Analysis of numba.cuda
======================

CUDA contains its own Python array type ``DeviceNDArray``.
It possibly corresponds to DPNP array type.

.. code-block:: python
  # numba/cuda/cudadrv/devicearray.py
  class DeviceNDArrayBase(_devicearray.DeviceArray):

  class DeviceNDArray(DeviceNDArrayBase):
    # like DPNP array
    @property
    def __cuda_array_interface__(self):

  class MappedNDArray(DeviceNDArrayBase, np.ndarray):
    # like USM shared memory
