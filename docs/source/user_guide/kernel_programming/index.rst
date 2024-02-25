.. _index:
.. include:: ./../../ext_links.txt

Kernel Programming
==================

The tutorial covers the most important features of the KAPI kernel programming
API and introduces the concepts needed to express data-parallel kernels in
numba-dpex.


Preliminary concepts
--------------------

Data parallelism
++++++++++++++++

Single Program Multiple Data
++++++++++++++++++++++++++++

Range v/s Nd-Range Kernels
++++++++++++++++++++++++++

Work items and Work groups
++++++++++++++++++++++++++

Basic concepts
--------------


Writing a *range* kernel
++++++++++++++++++++++++

A *range* kernel represents the simplest form of parallelism that can be
expressed in KAPI. A range kernel represents a data-parallel execution of the
same function by a set of work items. In KAPI, an instance of the
:py:class:`numba_dpex.kernel_api.Range` class represents the set of work items
and each work item in the ``Range`` is represented by an instance of the
:py:class:`numba_dpex.kernel_api.Item` class. As such these two classes are
essential to writing a range kernel in KAPI.

.. literalinclude:: ./../../../../numba_dpex/examples/kernel/vector_sum.py
   :language: python
   :lines: 8-9, 11-15
   :caption: **EXAMPLE:** A KAPI range kernel
   :name: ex_kernel_declaration_vector_sum

:ref:`ex_kernel_declaration_vector_sum` shows an example of a range kernel.
Every range kernel requires its first argument to be an ``Item`` and
needs to be launched via :py:func:`numba_dpex.experimental.launcher.call_kernel`
by passing an instance a ``Range`` object.

Do note that a ``Range`` object only controls the creation of work items, the
distribution of work and data over a ``Range`` still needs to be defined by the
user-written function. In the example, each work item access a single element of
each of the three array and performs a single addition operation. It is possible
to write the kernel differently so that each work item accesses multiple data
elements or conditionally performs different amount of work. The data access
patterns in a work item can have performance implications and programmers should
refer a more topical material such as the `oneAPI GPU optimization guide`_ to
learn more.

A range kernel is meant to express a basic `parallel-for` calculation that is
ideally suited for embarrassingly parallel kernels such as elementwise
computations over ndarrays. The API for expressing a range kernel does not
allow advanced features such as synchronization of work items and fine-grained
control over memory allocation on a device.

Writing an *nd-range* kernel
++++++++++++++++++++++++++++

The ``device_func`` decorator
+++++++++++++++++++++++++++++

Supported mathematical operations
+++++++++++++++++++++++++++++++++

Supported Python operators
++++++++++++++++++++++++++

Supported kernel arguments
++++++++++++++++++++++++++

Launching a kernel
++++++++++++++++++

Advanced topics
---------------

Local memory allocation
+++++++++++++++++++++++

Private memory allocation
+++++++++++++++++++++++++

Group barrier synchronization
+++++++++++++++++++++++++++++

Atomic operations
+++++++++++++++++

Async kernel execution
++++++++++++++++++++++

Specializing a kernel or a device_func
++++++++++++++++++++++++++++++++++++++
