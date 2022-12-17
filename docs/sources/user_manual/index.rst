.. _index:
.. include:: ./../ext_links.txt

.. image:: ./../_images/DPEP-large.png
    :width: 400px
    :align: center
    :alt: Data Parallel Extensions for Python

User Manual
===========

**Data Parallel Extension for Numba** extends `Numba*`_ capabilities to support heterogeneous computing.
It follows specifications of `SYCL*`_ standard in *pythonic* fashion with the goal of
providing a unified programming model for different device types.

The advantage of this approach is greatly improved code portability across *data-parallel devices*. The offered
programming model also accounts for amount of code changes required to adapt existing Python* scripts
to another *data-parallel device*.

These programming models can be split into three categories:

1. **Array-Style Programming** originates from the programming model inherited from `Numpy*`_. It works best
   when you have existing NumPy CPU script that you want to run on another *data-parallel device*.

2. **Direct Loops Programming** means coding explicit iterator to traverse through a countable set.
   Python has numerous ways to program iterators, from classic ``for`` and ``while`` loops to
   list comprehensions. Data Parallel Extension for Numba* can compile such constructs to a target device

3. **Kernel Function Programming** originates from GPU programming. CUDA*, OpenCl*, and SYCL* offer similar
   programming model known as the *data parallel kernel programming*. In this model you express the work in terms
   of *work items*. Data Parallel Extension for Numba* offers a way to write data parallel kernels in Python
   and compile them to a target device (GPU, CPU, data-parallel accelerators).

Table of Contents
-----------------

.. toctree::
    :maxdepth: 2

    getting_started
    auto_offload
    rng
    kernel_programming/index
    performance_tips
    troubleshooting
    useful_links
