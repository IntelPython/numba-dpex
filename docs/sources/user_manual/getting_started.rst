.. _getting_started:
.. include:: ./../ext_links.txt

.. |copy| unicode:: U+000A9

.. |trade| unicode:: U+2122

Getting Started
***************

Prerequisites and installation
==============================

1. Device drivers
-----------------

Since you are about to start programming data parallel devices beyond CPU, you will need an appropriate hardware.
In many cases your hardware comes with all necessary device drivers pre-installed.
But if you want the most up-to-date driver,
you can always `update it to the latest one <https://www.intel.com/content/www/us/en/download-center/home.html>`_.
Follow device driver installation instructions to complete this step, as needed.

2. Python interpreter
---------------------

You will need Python 3.8, 3.9, or 3.10 installed on your system. If you do not have one yet
the easiest way to do that is to install
`Intel Distribution for Python* <https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html>`_.
It will install all essential Python numerical and machine learning packages
optimized for Intel hardware, including **Data Parallel Extension for Numba**.
If you have Python installation from another vendor, it is fine too. All you need is to
install **Data Parallel Extension for Numba** manually.

3. Data Parallel Extension for Numba
--------------------------------------

You can skip this step if you already installed Intel Distribution for Python or Intel AI Analytics Toolkit,
because `Data Parallel Extensions for Python*`_ come pre-installed with these bundles.

If you do not plan using these bundles, you can use one of Python package managers to get
**Data Parallel Extension for Numba**.

Conda: ``conda install numba-dpex``

Pip: ``pip install numba-dpex``

The above commands will install the ``numba-dpex`` package, containing the **Data Parallel Extension for Numba**
along with its dependencies, including ``dpnp`` (`Data Parallel Extension for NumPy*`_),
``dpctl`` (`Data Parallel Control`_), and required compiler runtimes and drivers.

.. warning::

   Before installing with ``conda`` or ``pip`` it is strongly advised to update these package
   managers to latest versions

Hello, Data-Parallel World!
===========================

Since **Data Parallel Extension for Numba** is an extension for `Numba*`_, the assumption is
that you are already familiar how to program CPU with Numba.

.. seealso::

   For Getting Started experience with Numba please refer to
   `A ~5 minute guide to Numba <https://numba.readthedocs.io/en/stable/user/5minguide.html>`_

Porting existing Numba CPU program written using ``@njit`` to another *data-parallel device*
in many cases is fairly straightforward process:

    1. Substitute ``import numba`` with ``import numba_dpex``.
        This will enable ``numba`` backend capable of compiling for different `SYCL*`_
        devices.

    2. Substitute ``import numpy`` with ``import dpnp``
        This will enable array container capable of managing data on different data-parallel devices.
        `Data Parallel Extensions for Python*`_ implement the *compute-follows-data* paradigm,
        when the computation happens on the device where this data resides.

        Using ``dpnp`` you will be able to allocate data on the target device, and ``numba_dpex``
        will deduce the compilation target based on ``dpnp`` array inputs.

    3. In ``dpnp`` array creation or random number generation routines specify the target
       device as a keyword argument ``device=``.

To illustrate this process let's take an example of NumPy script that implements
the Monte Carlo method for estimation of :math:`\pi`:

.. literalinclude:: ./../../../numba_dpex/examples/hello_dpworld.py
    :caption: **EXAMPLE:** Original NumPy program
    :pyobject: hello_dpworld_numpy
    :lines: 3-
    :dedent: 4

In order to execute this script on a GPU (assuming you have one) the following changes will be required:

.. literalinclude:: ./../../../numba_dpex/examples/hello_dpworld.py
    :caption: **EXAMPLE:** Modified NumPy program to run on GPU
    :pyobject: hello_dpworld_dpnp
    :lines: 3-
    :emphasize-lines: 1, 4, 5
    :dedent: 4

First, we changed ``numpy`` to ``dpnp``. While ``numpy`` is close to data-parallel world, it is purely CPU
library. It is too generic to run on arbitrary data-parallel device, because its large part is handled by
interpreter, and interpreter can only run on the *host*. In contrast, ``dpnp`` implements
a subset of ``numpy`` that can be offloaded to data-parallel device.

Second, we need to specify on what device the input data will be allocated. In lines 4 and 5
we generate random data for arrays ``x`` and ``y``. This is our input data. Every ``dpnp`` array creation
and random number generation routine supports the ``device=`` keyword argument. In this example we specify
``device='gpu'`` which tells ``dpnp.random.random`` routine to generate the data on the default GPU device.

.. seealso::

    Refer to `Data Parallel Extension for Numpy*`_ documentation for details about device filter selection
    strings.

    Also, `Data Parallel Control`_ library provides advanced device management utilities that will,
    among other things, allow you to query the list of devices supported by your platform.

.. note::

    ``device=`` is an optional argument. If you omit it in ``dpnp`` array creation or random number
    generation routine, the default device will be selected.

    Default device is a platform specific. Typically if both CPU and GPU devices are present on a system,
    and all respective device drivers are properly installed, then the default device will be GPU.

Finally, `Data Parallel Extensions for Python*`_ implement the *compute-follows-data* paradigm of
heterogeneous computing. In this paradigm the actual computation happens where the data is allocated.

In our example we explicitly allocated our input data ``x`` and ``y`` on GPU device. Going forward,
according to the compute-follows-data, all computation that relies on ``x`` and ``y`` will be
performed on that GPU device:

1) In the array expression ``x * x + y * y <= 1.0`` the multiplication operation ``*`` will deduce from ``x``
   the device information and will perform the multiplication on that device. The result of this multiplication
   will be kept as a temporary array on that device too.

2) Similarly, another multiplication will deduce device information from ``y``, will perform multiplication
   on that device, and will store the result on that device too.

3) The ``+`` operation will deduce device information from two temporal arrays created
   on steps 1 and 2, will make sure that they
   belong to the same device, and will perform an addition on that device. The result will be stored
   in a temporal array on that device too.

4) Then the same logic will be applied to the operation ``<=``. And the boolean result will be
   kept on the device.

5) On the final step ``np.count_nonzero`` will again derive device information, will perform an operation
   on that device, and produce a "scalar" ``acc``.


Let's take one step further and modify our script so that part of ``dpnp`` code will be compiled
into a GPU offload kernel using ``numba-dpex``:

.. literalinclude:: ./../../../numba_dpex/examples/hello_dpworld.py
    :caption: **EXAMPLE:** Compile for GPU using Data Parallel Extension for Numba
    :pyobject: hello_dpworld_ndpx
    :lines: 3-
    :emphasize-lines: 2, 8-9, 11
    :dedent: 4

Like `Numba*`_ the Data Parallel Extension for Numba is just-in-time compiler. Rather than compiling
full Python script, Numba compiles user-selected functions. We wrap the part of the code that computes
the number of points within a unit circle into the function ``hello_numba_dpex()`` and decorate it
with ``njit``, which is a hint for the compiler to compile this function.

Why would someone need to compile this part of the program?

Remember previous exampple, where ``dpnp`` produced a number of temporary arrays? This is quite expensive.
Each of these arrays holds ``n = 50000`` elements. Large arrays create significant memory footprint, and
at some point you risk to run out of device memory. It is also not very good from a program performance
standpoint. Every device has "fast" memory but it is relatively small in size. With large number of temporary
arrays you're also risking to end up, when the device operations will be performed in "slow" memory.

This is where `Numba*`_ is really useful. Rather than performing multiple array operations on large data
the compiler can fuse these operations and perform these on smaller data chunks eliminating the need
to store large temporary data.
