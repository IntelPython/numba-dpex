.. include:: ./../ext_links.txt

Compiling and Offloading ``dpnp`` statements
============================================

Data Parallel Extension for NumPy* (``dpnp``) is a drop-in ``NumPy*``
replacement library built on top of oneMKL and SYCL. ``numba-dpex`` allows
various ``dpnp`` library function calls to be JIT-compiled using the
``numba_dpex.dpjit`` decorator. Presently, ``numba-dpex`` can compile several
``dpnp`` array constructors (``ones``, ``zeros``, ``full``, ``empty``), most
universal functions, ``prange`` loops, and vector expressions using
``dpnp.ndarray`` objects.

An example of a supported usage of ``dpnp`` statements in ``numba-dpex`` is
provided in the following code snippet:


.. ``numba-dpex`` implements its own runtime library to support offloading ``dpnp``
.. library functions to SYCL devices. For each ``dpnp`` function signature to be
.. offloaded, ``numba-dpex`` implements the corresponding direct SYCL function call
.. in the runtime and the function call is inlined in the generated LLVM IR.

.. code-block:: python

    import dpnp
    from numba_dpex import dpjit


    @dpjit
    def foo():
        a = dpnp.ones(1024, device="gpu")
        return dpnp.sqrt(a)


    a = foo()
    print(a)
    print(type(a))

.. :samp:`dpnp.ones(10)` will be called through |ol_dpnp_ones(...)|_.


.. Design
.. -------

.. ``numba_dpex`` uses the |numba.extending.overload| decorator to create a Numba*
.. implementation of a function that can be used in `nopython mode`_ functions.
.. This is done through translation of ``dpnp`` function signature so that they can
.. be called in ``numba_dpex.dpjit`` decorated code.

.. The specific SYCL operation for a certain ``dpnp`` function is performed by the
.. runtime interface. During compiling a function decorated with the ``@dpjit``
.. decorator, ``numba-dpex`` generates the corresponding SYCL function call through
.. its runtime library and injects it into the LLVM IR through
.. |numba.extending.intrinsic|_. The ``@intrinsic`` decorator is used for marking a
.. ``dpnp`` function as typing and implementing the function in nopython mode using
.. the `llvmlite IRBuilder API`_. This is an escape hatch to build custom LLVM IR
.. that will be inlined into the caller.

.. The code injection logic to enable ``dpnp`` functions calls in the Numba IR is
.. implemented by :mod:`numba_dpex.core.dpnp_iface.arrayobj` module which replaces
.. Numba*'s :mod:`numba.np.arrayobj`. Each ``dpnp`` function signature is provided
.. with a concrete implementation to generates the actual code using Numba's
.. ``overload`` function API. e.g.:

.. .. code-block:: python

..     @overload(dpnp.ones, prefer_literal=True)
..     def ol_dpnp_ones(
..         shape, dtype=None, order="C", device=None, usm_type="device", sycl_queue=None
..     ):
..         ...

.. The corresponding intrinsic implementation is in :file:`numba_dpex/dpnp_iface/_intrinsic.py`.

.. .. code-block:: python

..    @intrinsic
..    def impl_dpnp_ones(
..        ty_context,
..        ty_shape,
..        ty_dtype,
..        ty_order,
..        ty_device,
..        ty_usm_type,
..        ty_sycl_queue,
..        ty_retty_ref,
..    ):
..        ...

Parallel Range
---------------

``numba-dpex`` supports using the ``numba.prange`` statements with
``dpnp.ndarray`` objects. All such ``prange`` loops are offloaded as kernels and
executed on a device inferred using the compute follows data programming model.
The next examples shows using a ``prange`` loop.

.. implements the ability to run loops in parallel, the language
.. construct is adapted from Numba*'s ``prange`` concept that was initially
.. designed to run OpenMP parallel for loops. In Numba*, the loop-body is scheduled
.. in seperate threads, and they execute in a ``nopython`` Numba* context.
.. ``prange`` automatically takes care of data privatization. ``numba-dpex``
.. employs the ``prange`` compilation mechanism to offload parallel loop like
.. programming constructs onto SYCL enabled devices.

.. The ``prange`` compilation pass is delegated through Numba's
.. :file:`numba/parfor/parfor_lowering.py` module where ``numba-dpex`` provides
.. :file:`numba_dpex/core/parfors/parfor_lowerer.py` module to be used as the
.. *lowering* mechanism through
.. :py:class:`numba_dpex.core.parfors.parfor_lowerer.ParforLowerImpl` class. This
.. provides a custom lowerer for ``prange`` nodes that generates a SYCL kernel for
.. a ``prange`` node and submits it to a queue. Here is an example of a ``prange``
.. use case in ``@dpjit`` context:

.. code-block:: python

    import dpnp
    from numba_dpex import dpjit, prange


    @dpjit
    def foo():
        x = dpnp.ones(1024, device="gpu")
        o = dpnp.empty_like(a)
        for i in prange(x.shape[0]):
            o[i] = x[i] * x[i]
        return o


    c = foo()
    print(c)
    print(type(c))

.. Each ``prange`` instruction in Numba* has an optional *lowerer* attribute. The
.. lowerer attribute determines how the parfor instruction should be lowered to
.. LLVM IR. In addition, the lower attribute decides which ``prange`` instructions
.. can be fused together. At this point ``numba-dpex`` does not generate
.. device-specific code and the lowerer used is same for all device types. However,
.. a different :py:class:`numba_dpex.core.parfors.parfor_lowerer.ParforLowerImpl`
.. instance is returned for every ``prange`` instruction for each corresponding CFD
.. (Compute Follows Data) inferred device to prevent illegal ``prange`` fusion.

``prange`` loop statements can also be used to write reduction loops as
demonstrated by the following naive pairwise distance computation.

.. code-block:: python

  from numba_dpex import dpjit, prange
  import dpnp
  import dpctl


  @dpjit
  def pairwise_distance(X1, X2, D):
      """Naïve pairwise distance impl - take an array representing M points in N
      dimensions, and return the M x M matrix of Euclidean distances

      Args:
          X1 : Set of points
          X2 : Set of points
          D  : Outputted distance matrix
      """
      # Size of inputs
      X1_rows = X1.shape[0]
      X2_rows = X2.shape[0]
      X1_cols = X1.shape[1]

      float0 = X1.dtype.type(0.0)

      # Outermost parallel loop over the matrix X1
      for i in prange(X1_rows):
          # Loop over the matrix X2
          for j in range(X2_rows):
              d = float0
              # Compute exclidean distance
              for k in range(X1_cols):
                  tmp = X1[i, k] - X2[j, k]
                  d += tmp * tmp
              # Write computed distance to distance matrix
              D[i, j] = dpnp.sqrt(d)


  q = dpctl.SyclQueue()
  X1 = dpnp.ones((10, 2), sycl_queue=q)
  X2 = dpnp.zeros((10, 2), sycl_queue=q)
  D = dpnp.empty((10, 2), sycl_queue=q)

  pairwise_distance(X1, X2, D)
  print(D)


.. Fusion of Kernels
.. ------------------

.. ``numba-dpex`` can identify each NumPy* (or ``dpnp``) array expression as a
.. data-parallel kernel and fuse them together to generate a single SYCL kernel.
.. The kernel is automatically offloaded to the specified device where the fusion
.. operation is invoked. Here is a simple example of a Black-Scholes formula
.. computation where kernel fusion occurs at different ``dpnp`` math functions:

.. .. literalinclude:: ./../../../numba_dpex/examples/blacksholes_njit.py
..    :language: python
..    :pyobject: blackscholes
..    :caption: **EXAMPLE:** Data parallel kernel implementing the vector sum a+b
..    :name: blackscholes_dpjit


.. .. |numba.extending.overload| replace:: ``numba.extending.overload``
.. .. |numba.extending.intrinsic| replace:: ``numba.extending.intrinsic``
.. .. |ol_dpnp_ones(...)| replace:: ``ol_dpnp_ones(...)``
.. .. |numba.np.arrayobj| replace:: ``numba.np.arrayobj``

.. .. _low-level API: https://github.com/IntelPython/dpnp/tree/master/dpnp/backend
.. .. _`ol_dpnp_ones(...)`: https://github.com/IntelPython/numba-dpex/blob/main/numba_dpex/dpnp_iface/arrayobj.py#L358
.. .. _`numba.extending.overload`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-functions
.. .. _`numba.extending.intrinsic`: https://numba.pydata.org/numba-doc/latest/extending/high-level.html#implementing-intrinsics
.. .. _nopython mode: https://numba.pydata.org/numba-doc/latest/glossary.html#term-nopython-mode
.. .. _`numba.np.arrayobj`: https://github.com/numba/numba/blob/main/numba/np/arrayobj.py
.. .. _`llvmlite IRBuilder API`: http://llvmlite.pydata.org/en/latest/user-guide/ir/ir-builder.html
