.. _index:
.. include:: ./../../ext_links.txt

Programming kernel functions with Data Parallel Extension for Numba
===================================================================
`Data Parallel Extensions for Python*`_ introduce a concept of an *offload kernel*, defined as
a part of a Python program being submitted for execution to the device queue.

.. image:: ./../../_images/kernel-queue-device.png
    :scale: 50%
    :align: center
    :alt: Offload Kernel

There are multiple ways how to write offload kernels. CUDA*, OpenCl*, and SYCL* offer similar programming model
known as the *data parallel kernel programming*. In this model you express the work in terms of *work items*.
You split data into small pieces, and each piece will be a unit of work, or a *work item*. The total number of
work items is called *global size*. You can also group work items into bigger chunks called *work groups*.
The number of work items in the work group is called the *local size*. The following

Numba-dpex offers a way to write data-parallel kernels directly using Python.
The compiler extension to Numba has a programming model similar to the SYCL C++
domain-specific language. By providing similar abstractions as SYCL, Python
programmers can use the compiler to express data-parallelism using a
hierarchical syntax. Note that not all SYCL concepts are currently supported by
numba-dpex.

The explicit kernel programming mode of numba-dpex bears similarities with
Numba's other GPU backends: ``numba.cuda`` and ``numba.roc``. The documentation
should serves as a guide for using the current kernel programming features
available in numba-dpex.

.. toctree::
   :maxdepth: 2

   writing_kernels
   memory-management
   synchronization
   device-functions
   atomic-operations
   selecting_device
   memory_allocation_address_space
   reduction
   ufunc
   random
   supported-python-features
