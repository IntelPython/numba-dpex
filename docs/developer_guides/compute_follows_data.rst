Compute Follows Data
====================

Device context is deprecated.
Data defines an execution device.

When computation is offloaded to device the offloading device is defined by
device on which data is allocated.

Numba type should contain queue.
For each operation equivalent queue should be selected.
Computation should be queued to selected queue.

Native structure should contain field for queue.

.. code-block:: python
  queue = dpctl.equivalent_queue(a, b)
  if queue is None:
      raise ValueError
  queue.submit(kernel)

If arrays allocated on different devices (not equivalent) than raise an error
with recommendation to move the data explicitly.
It is by convention of Array API. See
`Device support <https://data-apis.org/array-api/latest/design_topics/device_support.html>`_.

There are convenient ways to move data explicitly to necessary device.
Moving implicitly sometimes introduce slowdowns which are also implisit.

.. code-block:: python

  b = move(b, a.queue)
  ...

Example of CFD - PyTorch.
Example of explicit - TensorFlow.

CuPy have agressive strategy to move (coerse) all data to device.
Data API recommends to use asarray() to coerse host data to coresponding array
object of the particular library (that target devices).

Numba offloading standard
`````````````````````````

DPNP array is the base for DPEX libraries (asarray() coerses to it).
In Numba function using DPNP means computations will be offloaded.

Results allocation
``````````````````

Results are allocated on the same device where computations are offloaded to.
