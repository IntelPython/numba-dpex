.. _programming_model:
.. include:: ./ext_links.txt

Programming Model
=================

This section describes the multiple facets of the programming model that defines
how programmers can use numba-dpex to develop parallel applications. The goal of
the section is to provide users new to accelerator programming or parallel
programming in general an introduction to some of the core concepts and map
those concepts to numba-dpex's interface.


Data-level parallelism
----------------------

A large part of the massive-level of parallelism offered by accelerators such as
GPUs is the ability to exploit *data-level parallelism* or simply *data
parallelism*. The term refers to a common pattern that occurs in many types of
programs where multiple units of the data accessed by the program can be
operated by a computer at the same time. All modern computing platforms offer
features to exploit data parallelism. Hardware features such as multiple nodes
of a cluster computer, multiple cores or execution units of a CPU or a GPU,
multiple threads inside a single execution unit, and even short-vector single
instruction multiple data (SIMD) registers on a core, all offer ways to exploit
data parallelism. Some of these hardware features such as SIMD registers are
exclusively designed for data parallelism, whereas others are more
general-purpose.

The diversity of the hardware landscape coupled with the different API required
by each type of hardware leads to conundrum for both programmers and programming
language designers: *How to define a common programming model that can express
data parallelism?* Defining a common programming model first and foremost
requires a common execution model backed by an operational semantics
:cite:p:`scott70` defining the computational steps of the execution model.


SPMD
----
logical abstraction

SIMD/SIMT implementation model


Execution Model
---------------

Memory Model
------------

Kernel Dependency Model
-----------------------

Compute follows data
--------------------

References
~~~~~~~~~~
.. bibliography::
