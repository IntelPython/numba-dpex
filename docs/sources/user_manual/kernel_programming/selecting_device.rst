.. _selecting_device:
.. include:: ./../../ext_links.txt

Compute-Follows-Data and Kernel Functions
==========================================
Kernel functions submission follows the `Compute-Follows-Data programming model
<https://intelpython.github.io/DPEP/main/heterogeneous_computing.html#compute-follows-data>`_.
It means the device information will be deduced from input array arguments passed to a kernel function.

In most cases you will be using `Data Parallel Extension for Numpy*`_ arrays as kernel function arguments.
However, ``numba-dpex`` will also work with any other tensor
implementation that follows the ``__sycl_usm_array_interface__`` protocol.
In all these cases ``numba-dpex`` will derive device queue
from respective fields of the ``__sycl_usm_array_interface__`` structure.

.. seealso::
   :ref:`SYCL_USM_ARRAY_INTERFACE` section of this document provides detailed specification
   of the protocol.

**Data Parallel Extensions for Numba** does not allow passing array arguments that
do not support ``__sycl_usm_array_interface__``. However, it does allow passing scalars.

At least one array argument supporting ``__sycl_usm_array_interface__`` is expected in a kernel function.
If more than one array argument is supplied then they must be associated with equivalent *device queues*.
The device queues are equivalent if they have the same **SYCL context**, and **SYCL device**, and the same
device **queue properties**.

All USM-types are accessible from device. Users can mix arrays
with different USM-type as long as they were allocated using
the equivalent SYCL queue.