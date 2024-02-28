.. _index:
.. include:: ./../ext_links.txt

Experimental Features
=====================

Numba-dpex includes various experimental features that are not yet suitable for
everyday production usage, but are included as an engineering preview.
The most prominent experimental features currently included in numba-dpex are
listed in this section.


Compiling and Offloading ``dpnp`` statements
--------------------------------------------

Data Parallel Extension for NumPy* (`dpnp`_) is a drop-in NumPy* replacement
library built using the oneAPI software stack including `oneMKL`_, `oneDPL`_ and
`SYCL*`_. numba-dpex has experimental support for compiling a subset of dpnp
functions. The feature is enabled by the :py:func:`numba_dpex.dpjit` decorator.

An example of a supported usage of dpnp in numba-dpex is provided in the
following code snippet:

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


Offloading ``prange`` loops
---------------------------

numba-dpex supports using the ``numba.prange`` statements with
``dpnp.ndarray`` objects. All such ``prange`` loops are offloaded as kernels and
executed on a device inferred using the compute follows data programming model.
The next examples shows using a ``prange`` loop.


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


Kernel fusion
-------------

.. ``numba-dpex`` can identify each NumPy* (or ``dpnp``) array expression as a
.. data-parallel kernel and fuse them together to generate a single SYCL kernel.
.. The kernel is automatically offloaded to the specified device where the fusion
.. operation is invoked. Here is a simple example of a Black-Scholes formula
.. computation where kernel fusion occurs at different ``dpnp`` math functions:
